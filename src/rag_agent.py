"""STEP 2 - grounded single-turn agent: query -> retrieve -> answer ONLY from the documents."""
import re

from . import llm
from .datatools import TODAY, TOOL_IMPLS, TOOL_SPECS
from .kb import catalog_labels, load_chunks
from .query import expand_query
from .records import Records
from .retriever import BM25Retriever
from .scoring import VALID_DECISIONS
from .voice import tts_language   # script/language detection only (no Gnani call)

# Documents that always apply, so they are loaded into every prompt instead of searched for:
# knowledge-base rules (precedence, today's date, business days) + style guide + escalation rules.
PINNED_DOCS = ("KB-00", "AGT-STY", "AGT-ESC")

SYSTEM_TEMPLATE = """You are ZentroMart's AI customer-support assistant. ZentroMart is an Indian online store selling groceries and electronics. Today is 20 October 2026.

HOW YOU MUST ANSWER
1. Use ONLY the DOCUMENT EXCERPTS and DATABASE RECORDS given in the message, plus the always-on rules below. Never use outside knowledge for policies, amounts, time limits or dates, and never invent them.

2. Choose the decision type:
 - "answer": you can give the customer a useful answer from the documents. This includes a clear "no" (for example: not returnable), and includes questions the documents do not cover: then say plainly that you don't have that information and offer to connect a human colleague. A general policy question never needs an order ID: answer it with the general rule and mention any special condition that depends on order details (for example a sale-period window). If you give useful steps or facts, use "answer" even if you offer more help afterwards.
 - "clarify": ONLY when you cannot answer without one missing detail that matters (for example which payment method was used, or which item). Ask ONE short question.
 - "escalate": when an escalation rule applies. Check every rule in the escalation guide against the customer's message and the database record, and apply a rule only if ALL parts of its condition are met (for example a high-value rule needs a damaged, defective or wrong item AND a price above the limit; it does not apply to warranty or other questions). Escalation beats clarify: do not ask for photos or details first. Tell the customer that a human colleague will contact them. You cannot create tickets or take actions in this mode, so never say that you created a ticket or did anything.
 - "refuse": ONLY when the customer demands something the policy does not allow (an exception, compensation, another customer's personal data). A plain "no, this is not eligible" to an eligibility question is "answer", not "refuse".

3. If excerpts conflict, follow the precedence rules. A SUPERSEDED document applies only to the cases its warning says it still covers.

4. Dates: never work out or guess a date yourself. For a refund timeline of "MIN to MAX business days", call business_day_range ONCE with the day the refund started and both numbers from the policy. For a single fixed number of business days use add_business_days. For return deadlines use add_calendar_days from the delivery date. Give the earliest and the latest date, using ONLY dates returned by the tools. "Yesterday" and "day before yesterday" are relative to today.

5. If the question is about a product, use its DATABASE RECORD (return category, price, fulfilled_by, assured, warranty_months).

6. Include the specific facts the question needs: numbers, time periods (for example warranty months), fees, deadlines, the condition behind a "no" (for example that only unopened items with the seal intact can be returned) and where to act (for example "My Orders" in the app). Write numbers as digits (3 attempts, 7 days, 49 rupees). Write every date with day, month name and year, for example 27 October 2026. Do not mention section IDs such as REF-1 in the reply. Do not invent buttons, steps or details that are not in the documents.

7. Language and script: reply in the customer's language AND script. Hindi in Devanagari -> Hindi in Devanagari. Hinglish (Hindi words written in Roman letters) -> Hinglish in Roman letters, never Devanagari. English -> English. Keep app names such as "My Orders" in English.

8. Follow the style rules below (opening, length, one question at a time).

OUTPUT: reply with ONLY one JSON object and nothing else:
{{"decision": "answer" | "clarify" | "escalate" | "refuse", "reply": "<what you say to the customer>", "sources": ["<section ids your answer relies on, e.g. REF-2 or products.csv:ELE-0008>"]}}

ALWAYS-ON RULES FROM THE KNOWLEDGE BASE
{pinned}
"""

REWRITE_PROMPT = ("Rewrite the customer's message (Hindi, Hinglish or English) as a SHORT English keyword query "
                  "(max 15 words) for searching a store's policy documents about returns, refunds, delivery, "
                  "cancellations, warranty, payments, damaged items, membership and escalation. "
                  "Keep product names, SKUs, dates and numbers. Output only the query.")


# STY-1: "Introduce yourself as ZentroMart's AI support assistant in your first reply."
# Models skip this sometimes, so it is enforced in code (in the customer's language and script).
_INTRO = {"en-IN": "I'm ZentroMart's AI support assistant.",
          "hi-en": "Main ZentroMart ka AI support assistant hoon.",
          "hi-IN": "मैं ZentroMart का AI सपोर्ट असिस्टेंट हूँ।"}
_GREETING = re.compile(r"^\s*(hello|hi|hey|namaste|namaskar|नमस्ते|हैलो)\b[!,.\s]*", re.I)


def ensure_intro(reply: str, first_reply: bool, lang: str = None) -> str:
    if not first_reply or not reply.strip():
        return reply
    low = reply.lower()
    if re.search(r"\bai\b", low) and ("assistant" in low or "असिस्टेंट" in reply or "सहायक" in reply):
        return reply                                           # already introduced itself
    intro = _INTRO[lang or tts_language(reply)]
    m = _GREETING.match(reply)
    return (reply[:m.end()] + intro + " " + reply[m.end():]) if m else f"{intro} {reply}"


class RagAgent:
    def __init__(self, k: int = 8, rewrite: str = "auto", model: str = None):
        assert rewrite in ("auto", "always", "never")
        self.k, self.rewrite_mode, self.model = k, rewrite, model
        self.chunks = load_chunks()
        self.retriever = BM25Retriever(self.chunks)
        self.records = Records()
        self.labels = catalog_labels(self.chunks)
        self.pinned = [c for c in self.chunks if c.doc_id in PINNED_DOCS]
        self.pinned_ids = {c.id for c in self.pinned}
        self.system_prompt = SYSTEM_TEMPLATE.format(pinned="\n\n".join(c.render() for c in self.pinned))

    # ---- retrieval -------------------------------------------------------------
    def _rewrite(self, question: str) -> str:
        non_ascii = any(ord(ch) > 127 and ch not in "\u2019\u2018\u201c\u201d\u20b9" for ch in question)
        if self.rewrite_mode == "never" or (self.rewrite_mode == "auto" and not non_ascii):
            return ""
        try:
            out = llm.chat([{"role": "system", "content": REWRITE_PROMPT},
                            {"role": "user", "content": question}],
                           model=llm.REWRITE_MODEL, max_tokens=400, reasoning_effort="low")
            return re.sub(r"\s+", " ", out).strip()
        except Exception:
            return ""

    def retrieve(self, question: str, history_text: str = "", extra_products=None, hints: str = "") -> dict:
        """history_text / extra_products / hints let a conversation add earlier turns, the products of the
        order and facts of the order (payment method, Plus ...) to the search. Defaults = single question."""
        prods = (self.records.find_products(question) or self.records.find_products(history_text)
                 or list(extra_products or []))
        pins = self.records.find_pincodes(question)
        rewrite = self._rewrite(question)
        query = expand_query(f"{question} {history_text} {rewrite} {hints}".strip(), prods, pins, self.labels)
        hits = self.retriever.search(query, k=self.k, exclude_ids=self.pinned_ids)
        return {"products": prods, "pincodes": pins, "rewrite": rewrite, "query": query,
                "chunks": [c for c, _ in hits]}

    # ---- answering -------------------------------------------------------------
    def _user_message(self, question, ctx, channel, first_reply):
        records = [self.records.render_product(p) for p in ctx["products"]] + \
                  [self.records.render_pincode(p) for p in ctx["pincodes"]]
        parts = [f"CUSTOMER MESSAGE (channel: {channel}; "
                 f"{'first message of the conversation' if first_reply else 'follow-up'}):\n{question}",
                 "DATABASE RECORDS:\n" + ("\n".join(records) if records else "(none found for this message)"),
                 "DOCUMENT EXCERPTS (most relevant first):\n" +
                 "\n\n".join(c.render() for c in ctx["chunks"])]
        return "\n\n".join(parts)

    def answer(self, question: str, channel: str = "chat", first_reply: bool = True) -> dict:
        ctx = self.retrieve(question)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._user_message(question, ctx, channel, first_reply)}]
        raw, trace = llm.chat_with_tools(messages, TOOL_SPECS, TOOL_IMPLS, model=self.model)
        data = llm.parse_json(raw)
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            decision = "answer"
        if not str(data.get("reply") or raw).strip():      # model produced nothing: never send silence
            data = {"reply": "Sorry, I could not process that just now. Could you please ask again?"}
            decision = "clarify"
        sources = data.get("sources") or []
        return {
            "decision": decision,
            "reply": ensure_intro(str(data.get("reply") or raw).strip(), first_reply),
            "sources": [str(s) for s in sources] if isinstance(sources, list) else [str(sources)],
            "retrieved_ids": [c.id for c in ctx["chunks"]],
            "context_ids": sorted(self.pinned_ids) + [c.id for c in ctx["chunks"]]
                           + [f"products.csv:{p['sku']}" for p in ctx["products"]],
            "search_query": ctx["query"],
            "rewrite": ctx["rewrite"],
            "tool_calls": trace,
        }

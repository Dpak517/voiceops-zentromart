"""STEP 3 - conversation agent: memory, caller verification, order lookup, tickets, handover.

The LLM runs the conversation. Code enforces what a prompt cannot guarantee:
  * one retry on failed verification, then the order is locked (IDV-3)
  * a verification value must really have been said by the customer
  * order details are shown to the model only after verification
  * at most one ticket per conversation, and only claimed when created=True
"""
import json
import re
from dataclasses import dataclass, field

from . import llm
from .datatools import TOOL_IMPLS as DATE_IMPLS
from .datatools import TOOL_SPECS as DATE_SPECS
from .kb import catalog_labels, load_chunks
from .query import expand_query
from .rag_agent import ensure_intro
from .records import Records
from .retriever import BM25Retriever
from .scoring import VALID_DECISIONS

PINNED_DOCS = ("KB-00", "AGT-STY", "AGT-ESC", "AGT-IDV")
MAX_HISTORY_MESSAGES = 12

SYSTEM_TEMPLATE = """You are ZentroMart's AI customer-support assistant. ZentroMart is an Indian online store selling groceries and electronics. Today is 20 October 2026. You are in a conversation with a customer (chat or phone call).

HOW YOU MUST WORK
1. Use ONLY the DOCUMENT EXCERPTS, DATABASE RECORDS, the verified ORDER RECORD, tool results and the always-on rules below. Never use outside knowledge for policies, amounts, time limits or dates, and never invent them. If the documents do not cover a question, say plainly that you don't have that information and offer a human colleague (decision "answer"). Never ask for an order ID just to answer a general policy question.

2. ORDER HELP FLOW
 a. If the customer needs help with their own order and no order ID is known yet, ask for the order ID (decision "clarify"). Customers may say it in groups of digits: "8978 2837" means 89782837.
 b. When you get an order ID, call lookup_order with the order_id ONLY. Then ask for ONE of: the last 4 digits of the registered phone number, or the delivery pincode (decision "clarify"). Mention both options. Say nothing about the order before verification.
 c. When the customer gives 4 digits (phone) or a 6-digit pincode, call lookup_order again with order_id AND verification_value. Use only values the customer actually said.
 d. If verified: say so briefly. If the customer already described a problem or question earlier in the conversation, answer it right away from the ORDER RECORD and documents. Otherwise ask how you can help. Use decision "answer".
 e. If verification fails the FIRST time: say the details did not match and ask the customer to check and try again with the last 4 digits or the pincode (decision "clarify"). If it fails a SECOND time, or the tool says locked: share NO order details; say you cannot share details because verification failed, suggest contacting support from the registered phone number, and offer general policy help (decision "refuse"). Never reveal the correct value.
 f. General policy questions can be answered at any time without verification. Questions about the customer's own order need a verified order.
 g. Never read out a full phone number, email, street address or pincode.

3. DECISION TYPES
 - "answer": a useful answer from the documents or order record, including a clear "no", verification success, and "I don't have that information". If you give useful facts or steps, use "answer" even if you offer more help.
 - "clarify": only when you cannot proceed without one missing detail (order ID, verification value, which item, how they paid). Ask ONE short question.
 - "escalate": when an escalation rule applies. Check every rule in the escalation guide against the customer's message and the ORDER RECORD (for example contacts_last_7_days), and apply a rule only if ALL parts of its condition are met (a high-value rule needs a damaged, defective or wrong item AND a price above the limit). Escalation beats clarify. FIRST call create_ticket (summary: the customer's issue, detected intent, order ID, what the customer already told you, what you already checked, and the reason for escalation), THEN tell the customer a human colleague will contact them. Say a ticket was created only if create_ticket returned created=true; if it failed, say you could not create it just now. Never promise outcomes, amounts or times beyond what a policy states.
 - "refuse": when the customer demands something policy does not allow: compensation, a voucher or any exception, cancelling an order that has already shipped, approving a claim yourself, or another person's data. Apologise briefly, explain what the policy allows and what you can do. Do not create a ticket for a plain refusal.
 You cannot cancel orders, approve claims or issue refunds yourself. To cancel an order that has not shipped, guide the customer to My Orders in the app. Never say you did something you did not do.

4. If excerpts conflict, follow the precedence rules. A SUPERSEDED document applies only to the cases its warning says it still covers.

5. Dates: never work out or guess a date yourself. For "MIN to MAX business days" timelines call business_day_range once; for return deadlines call add_calendar_days from the delivered_on date using the window the documents give for that item's category and order date; business_days_between helps to check whether a refund is overdue. Use only dates the tools return. "Yesterday" is relative to today.

6. Use the ORDER RECORD for status, dates, payment method, items, fees and refund fields. If the order has several items, use the one the customer names; if it is unclear, ask which one.

7. Remember the conversation. A follow-up such as "how long will that take?" or "and the refund?" refers to the previous topic and order. Never ask again for details the customer already gave.

8. Include the specific facts the question needs (numbers, periods, fees, deadlines, the condition behind a "no", where to act such as "My Orders" in the app). Write numbers as digits. Write every date with day, month name and year, for example 27 October 2026. Do not mention section IDs such as REF-1. Do not invent buttons, steps or details that are not in the documents.

9. Reply in the customer's language AND script: Hindi in Devanagari -> Hindi; Hinglish (Hindi in Roman letters) -> Hinglish in Roman letters, never Devanagari; English -> English. If the customer switches language, switch with them. Keep app names such as "My Orders" in English. Follow the style rules below.

OUTPUT: reply with ONLY one JSON object and nothing else:
{{"decision": "answer" | "clarify" | "escalate" | "refuse", "reply": "<what you say to the customer>", "sources": ["<section ids your answer relies on, e.g. REF-2>"]}}

ALWAYS-ON RULES FROM THE KNOWLEDGE BASE
{pinned}
"""

ORDER_TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up an order. First call it with order_id ONLY: it reveals nothing and tells you to ask the "
                       "customer for proof. After the customer says the last 4 digits of the registered phone number "
                       "or the 6-digit delivery pincode, call it again with that value as verification_value.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "digits only, e.g. 89782837"},
            "verification_value": {"type": "string", "description": "4-digit phone suffix or 6-digit pincode said by the customer"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Create a human handover ticket. Call it BEFORE telling the customer a colleague will contact them.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "issue, detected intent, order ID, what the customer told you, what you checked, reason for escalation"},
            "reason": {"type": "string"}, "intent": {"type": "string"}, "order_id": {"type": "string"}},
            "required": ["summary", "reason", "intent"]}}},
]

_DROP_KEYS = {"customer_id", "delivery_pincode", "found", "verified"}


def _digits(x) -> str:
    return re.sub(r"\D", "", str(x or ""))


_WORD_DIGITS = {"zero": "0", "shunya": "0", "one": "1", "ek": "1", "two": "2", "do": "2", "three": "3", "teen": "3",
                "four": "4", "char": "4", "chaar": "4", "five": "5", "panch": "5", "paanch": "5",
                "six": "6", "chhe": "6", "chhah": "6", "seven": "7", "saat": "7", "eight": "8", "aath": "8",
                "nine": "9", "nau": "9"}


def spoken_digits(text: str) -> str:
    """Digits a customer said, as numerals OR words ('four five three three'). Non-numbers break a run."""
    out = []
    for tok in re.findall(r"[A-Za-z]+|\d+", str(text).lower()):
        out.append(tok if tok.isdigit() else _WORD_DIGITS.get(tok, "|"))
    return "".join(out)


def slim_record(rec: dict) -> dict:
    """Order record as the model sees it: no nulls, no customer id, no pincode."""
    out = {k: v for k, v in rec.items() if v is not None and k not in _DROP_KEYS and k != "items"}
    out["items"] = [{k: it.get(k) for k in ("sku", "title", "return_category", "qty", "unit_price_inr")}
                    for it in rec.get("items", [])]
    return out


@dataclass
class State:
    history: list = field(default_factory=list)      # plain {"role","content"} turns
    user_texts: list = field(default_factory=list)
    order_id: str = ""
    verified: bool = False
    failures: int = 0
    record: dict = None
    ticket: dict = None
    turn: int = 0


class ConversationAgent:
    def __init__(self, k: int = 6, rewrite: str = "never", model: str = None):
        assert rewrite in ("auto", "always", "never")
        self.k, self.rewrite_mode, self.model = k, rewrite, model
        self.chunks = load_chunks()
        self.retriever = BM25Retriever(self.chunks)
        self.records = Records()
        self.labels = catalog_labels(self.chunks)
        self.pinned = [c for c in self.chunks if c.doc_id in PINNED_DOCS]
        self.pinned_ids = {c.id for c in self.pinned}
        self.system_prompt = SYSTEM_TEMPLATE.format(pinned="\n\n".join(c.render() for c in self.pinned))
        self.impls = {**DATE_IMPLS, "lookup_order": self._lookup_order, "create_ticket": self._create_ticket}
        self.tools = DATE_SPECS + ORDER_TOOL_SPECS
        self.state = State()

    def reset(self):
        self.state = State()

    # ---------------------------------------------------------------- tools (with guards)
    @staticmethod
    def _real_tools():
        from tools import create_ticket, lookup_order     # the provided mock tools (tools/tools.py)
        return lookup_order, create_ticket

    def _lookup_order(self, order_id, verification_value=None):
        st, real_lookup = self.state, self._real_tools()[0]
        oid = _digits(order_id)
        proof = _digits(verification_value) if verification_value not in (None, "") else None
        if st.verified and oid == st.order_id:
            return {"already_verified": True,
                    "message": "Customer is already verified for this order. Use the ORDER RECORD in the message."}
        if oid != st.order_id:                              # a different order: start verification again
            st.order_id, st.verified, st.failures, st.record = oid, False, 0, None
        if st.failures >= 2:
            return {"verified": False, "locked": True,
                    "message": "Verification failed twice. Share NO order details. Say you cannot share them, "
                               "suggest contacting support from the registered phone number, offer general policy help."}
        if proof is not None and proof not in spoken_digits(" ".join(st.user_texts)):
            return {"error": "That verification value was not said by the customer. Ask the customer for it."}
        res = real_lookup(oid, proof)
        if res.get("verified") is True and res.get("found"):
            st.verified, st.record = True, slim_record(res)
            return {"verified": True, "order_record": st.record}
        if proof is None:                                    # step 1: nothing revealed, proof needed
            return {"verification_required": True,
                    "message": "Ask the customer for the last 4 digits of the registered phone number OR the delivery pincode."}
        st.failures += 1
        if st.failures >= 2:
            return {"verified": False, "locked": True,
                    "message": "Verification failed a second time. Share NO order details. Say you cannot share them, "
                               "suggest contacting support from the registered phone number, offer general policy help."}
        return {"verified": False, "attempt": st.failures,
                "message": "Details did not match. Ask the customer to check and try again (one retry allowed). "
                           "Never reveal the correct value."}

    def _create_ticket(self, summary, reason=None, intent=None, order_id=None):
        st = self.state
        if not summary or len(str(summary).strip()) < 25:      # the model's own text must be meaningful (ESC-3)
            return {"created": False,
                    "error": "summary too short: include the issue, the intent, what the customer told you, "
                             "what you checked and the reason for escalation"}
        if st.ticket and st.ticket.get("created"):
            return {"created": True, "ticket_id": st.ticket["ticket_id"], "note": "ticket already created in this conversation"}
        oid = _digits(order_id) or st.order_id or None
        extra = f" | order_id={oid or 'n/a'} | verified={st.verified}"
        if st.record:
            extra += f" | contacts_last_7_days={st.record.get('contacts_last_7_days')} | order_status={st.record.get('status')}"
        res = self._real_tools()[1](summary=f"{summary}{extra}", order_id=oid, reason=reason, intent=intent)
        if res.get("created"):
            st.ticket = res
        return res

    # ---------------------------------------------------------------- retrieval
    @staticmethod
    def _id_or_proof_only(text: str) -> bool:
        """'Order number hai 8978 2837' / 'last four digits 4533': no documents needed for these turns."""
        squeezed = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        return bool(re.search(r"\d{4,}", squeezed)) and len(re.findall(r"[^\W\d_]", text)) <= 40

    def retrieve(self, text: str) -> dict:
        st = self.state
        if not st.verified and self._id_or_proof_only(text):
            return {"products": [], "chunks": [], "query": "", "skipped": True}
        query = text
        if len(text.split()) <= 8:                           # short follow-up: add what we were talking about
            query = " ".join(st.user_texts[-3:-1] + [text])
        if st.record and st.record.get("is_plus"):
            query += " ZentroMart Plus"
        prods = self.records.find_products(query)
        q = expand_query(query, prods, self.records.find_pincodes(query), self.labels)
        hits = self.retriever.search(q, k=self.k, exclude_ids=self.pinned_ids)
        return {"products": prods, "chunks": [c for c, _ in hits], "query": q, "skipped": False}

    # ---------------------------------------------------------------- one turn
    def _user_message(self, text, ctx, channel):
        st = self.state
        parts = [f"CUSTOMER MESSAGE (channel: {channel}; turn {st.turn + 1}):\n{text}",
                 f"SESSION STATE: order_id={st.order_id or 'none yet'}; verified={st.verified}; "
                 f"failed verification attempts={st.failures}; ticket_created={bool(st.ticket)}"]
        if st.verified and st.record:
            parts.append("ORDER RECORD (verified; never read out raw ids, phone, email or address):\n"
                         + json.dumps(st.record, ensure_ascii=False))
        prod_lines = {p["sku"]: self.records.render_product(p) for p in ctx["products"]}
        if st.verified and st.record:
            for it in st.record.get("items", []):
                p = self.records.by_sku.get(it["sku"])
                if p:
                    prod_lines.setdefault(p["sku"], self.records.render_product(p))
        parts.append("DATABASE RECORDS:\n" + ("\n".join(prod_lines.values()) or "(none)"))
        parts.append("DOCUMENT EXCERPTS (most relevant first):\n" +
                     ("\n\n".join(c.render() for c in ctx["chunks"]) if ctx["chunks"]
                      else "(not needed for this message)"))
        return "\n\n".join(parts)

    def answer(self, text: str, channel: str = "chat", first_reply: bool = None) -> dict:
        st = self.state
        first = (st.turn == 0) if first_reply is None else first_reply
        st.user_texts.append(text)
        ctx = self.retrieve(text)
        messages = ([{"role": "system", "content": self.system_prompt}] + st.history[-MAX_HISTORY_MESSAGES:]
                    + [{"role": "user", "content": self._user_message(text, ctx, channel)}])
        raw, trace = llm.chat_with_tools(messages, self.tools, self.impls, model=self.model)
        data = llm.parse_json(raw)
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            decision = "answer"
        reply = str(data.get("reply") or raw).strip()
        if not reply:
            reply, decision = "Sorry, I could not process that just now. Could you please say it again?", "clarify"
        reply = ensure_intro(reply, first)
        st.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        st.turn += 1
        sources = data.get("sources") or []
        return {
            "decision": decision, "reply": reply,
            "sources": [str(s) for s in sources] if isinstance(sources, list) else [str(sources)],
            "retrieved_ids": [c.id for c in ctx["chunks"]],
            "context_ids": sorted(self.pinned_ids) + [c.id for c in ctx["chunks"]]
                           + [f"products.csv:{p['sku']}" for p in ctx["products"]],
            "search_query": ctx["query"], "rewrite": "", "tool_calls": trace,
            "state": {"order_id": st.order_id, "verified": st.verified, "failures": st.failures,
                      "ticket_id": (st.ticket or {}).get("ticket_id", "")},
        }

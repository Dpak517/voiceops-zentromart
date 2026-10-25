"""Build report.pdf (max 4 pages) from YOUR result files. Missing files are reported as 'not measured'.

    python make_report.py --team "Team Name" --name "Your Name" --model "openai/gpt-oss-120b"
Optional: --step1 outputs/step1_guess.csv --step1-cautious outputs/step1_cautious.csv
          --step2 outputs/step2_results.csv --step3 outputs/step3_results.csv
Manual override if a CSV was overwritten:  --s1 8/52  --s1c 10/52  --s2 38/52  --s3 60/96
"""
import argparse
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def load(path):
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return None


def score(df, override):
    if override:
        a, b = override.split("/")
        return int(a), int(b)
    if df is None or "passed" not in df.columns or len(df) == 0:
        return None
    return int((df["passed"] == "True").sum()), len(df)


def fmt(s):
    if s is None:
        return "not measured"
    return f"{s[0]}/{s[1]} = {100 * s[0] / s[1]:.1f}%" + (" (PARTIAL RUN - rerun or use --s1/--s2)" if s[1] < 40 else "")


def mean_col(df, col):
    if df is None or col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if v.empty else float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Team")
    ap.add_argument("--name", default="")
    ap.add_argument("--model", default="openai/gpt-oss-120b (Groq)")
    ap.add_argument("--step1", default="outputs/step1_guess.csv")
    ap.add_argument("--step1-cautious", default="outputs/step1_cautious.csv")
    ap.add_argument("--step2", default="outputs/step2_results.csv")
    ap.add_argument("--step3", default="outputs/step3_results.csv")
    for k in ("s1", "s1c", "s2", "s3"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default="report.pdf")
    a = ap.parse_args()

    d1, d1c, d2, d3 = load(a.step1), load(a.step1_cautious), load(a.step2), load(a.step3)
    s1, s1c, s2, s3 = score(d1, a.s1), score(d1c, a.s1c), score(d2, a.s2), score(d3, a.s3)
    rec_s, rec_c = mean_col(d2, "recall_search"), mean_col(d2, "recall_context")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=11.5, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=6, spaceAfter=2)
    bul = ParagraphStyle("bul", parent=body, leftIndent=11, bulletIndent=2)
    P = lambda t: Paragraph(t, body)
    B = lambda t: Paragraph(t, bul, bulletText="\u2022")

    def table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF1")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    f = []
    f.append(Paragraph("VoiceOps 2026 - ZentroMart voice support agent", h1))
    f.append(P(f"<b>{a.team}</b> {('- ' + a.name) if a.name else ''} &nbsp;|&nbsp; LLM: {a.model} &nbsp;|&nbsp; "
               "Speech: Gnani STT + TTS only &nbsp;|&nbsp; Data: the synthetic ZentroMart dataset only."))

    f.append(Paragraph("1. What I built", h2))
    f.append(P("One agent that answers from company documents, holds a conversation, and speaks. Pipeline for one turn: "
               "customer audio -> Gnani STT -> agent (LLM + retrieval + tools) -> Gnani TTS -> spoken reply. "
               "Each step reuses the previous one, and every answer logs the document chunks it used."))

    f.append(Paragraph("2. Step 1 - prompt-only baseline", h2))
    f.append(P("A system prompt makes the LLM ZentroMart's assistant (role, limits, introduces itself as an AI assistant) "
               "and returns JSON {decision, reply}. It never sees the documents. I tried two prompts on the 52 practice "
               "questions: a <i>cautious</i> one (say 'I don't know' and offer a human when unsure) and a <i>guessing</i> one "
               "(always answer from general knowledge). Scoring uses the dataset's own rules: expected decision, "
               "must_include (';' = AND, ' / ' = OR) and must_not_include."))
    f.append(table([["Run", "Result on 52 practice questions"],
                    ["Step 1, cautious prompt", fmt(s1c)], ["Step 1, guessing prompt", fmt(s1)],
                    ["Step 2, RAG", fmt(s2)]], [6 * cm, 10.5 * cm]))
    f.append(Spacer(1, 3))
    f.append(P("The cautious prompt refuses or escalates almost everything; the guessing prompt invents policy numbers. "
               "Both fail for the same reason: the model has no access to ZentroMart's documents."))

    f.append(Paragraph("3. Step 2 - grounded answers (RAG)", h2))
    for t in ["<b>Chunking:</b> each '## [SECTION-ID]' section of kb/*.md is one chunk, keeping doc type, status and "
              "effective date. Superseded and marketing text are down-ranked and labelled so precedence (policy > guide > help > FAQ > marketing) "
              "is applied; the superseded v3 returns policy is still used for electronics delivered before 1 August 2026.",
              "<b>Retrieval:</b> BM25 over 84 chunks, plus query expansion that bridges customer words to document words "
              "('send it back' -> 'return'), hints from the product record (category, marketplace seller, price) and an LLM rewrite "
              "of Hindi questions into English keywords. Rules that always apply (style guide, escalation rules, business-day "
              "and date rules) are loaded into every prompt.",
              "<b>Records and tools:</b> products.csv rows are added when a product or SKU is mentioned. A date calculator tool "
              "does business-day (Mon-Sat, holidays excluded) and calendar-day arithmetic, but policy numbers still come from the documents, "
              "so nothing is hard-coded.",
              "<b>Honesty rule:</b> the prompt tells the model to say 'I don't have that information' and offer a human when the documents do not cover a question."]:
        f.append(B(t))
    parts = [f"Answer score: {fmt(s2)}."]
    if rec_s is not None:
        parts.append(f"Gold-section recall: {100 * rec_s:.0f}% from search alone, {100 * (rec_c or rec_s):.0f}% including the always-loaded rules.")
    f.append(P(" ".join(parts)))

    f.append(Paragraph("4. Step 3 - conversation", h2))
    for t in ["<b>Memory:</b> the last turns, the order ID, verification state and the verified order record are kept, so "
              "'and how long will the refund take?' is understood without repeating anything.",
              "<b>Verification (code-enforced):</b> order ID first (spoken groups like '8978 2837' are normalised), then last 4 phone digits or pincode. "
              "One retry, then the order is locked and no details are shared. A verification value the customer never said "
              "(including spoken digits) is rejected; the pincode and customer ID are never shown to the model.",
              "<b>Tools:</b> the provided lookup_order and create_ticket. A ticket is created once per conversation, only with a meaningful "
              "summary, and the agent says it exists only if created=True. Compensation demands and cancelling shipped orders are refused politely without a ticket; "
              "the agent never claims to have cancelled or approved anything.",
              "<b>Handover:</b> human requested, repeat contact (contacts_last_7_days), high-value damage, fraud, legal threats and safety risks create a ticket and hand over."]:
        f.append(B(t))
    t3 = ("Test harness: replaying all 23 practice conversations (96 turns) with a scripted model passed 96/96. This validates the tools, verification, ticket creation "
          "and scoring code, not the model's judgement. ")
    t3 += (f"Real-model run on the practice conversations: {fmt(s3)}." if s3 else
           "A full real-model run was not completed because of the Groq free-tier daily token limit.")
    f.append(P(t3))

    f.append(Paragraph("5. Step 4 - voice", h2))
    f.append(P("Gnani STT (language en-IN by default, which accepts English mixed with Hindi) feeds the agent; replies are spoken with Gnani TTS "
               "(model named explicitly). The TTS language follows the reply: Devanagari -> hi-IN, Roman Hindi -> hi-en, otherwise en-IN. Replies are rewritten for the ear: "
               "amounts as 'rupees', long order numbers read in groups of four digits, markdown and section tags removed. Clips over 55 s are split. "
               "Silence or unclear audio gets a spoken 'Sorry, I didn't catch that', and a failure in the agent still produces a spoken apology, "
               "so every audio file gets a reply. One command processes a folder: python run.py --audio-dir DIR --out-dir sample_outputs."))

    f.append(Paragraph("6. What went wrong and what I fixed", h2))
    for t in ["<b>Empty replies</b> on date questions (reasoning used all tokens): retry with lower reasoning effort and a bigger budget; never send silence.",
              "<b>Half-done date maths:</b> the model asked the calculator only for the minimum days and guessed the rest. Fixed with one range tool that returns earliest and latest date.",
              "<b>Over- and under-escalation:</b> a warranty question on a >10,000 INR item was escalated (my own query hint pulled in the damage rule); a cracked laptop was not. Fixed by "
              "adding the hint only for damage questions and by rules saying each escalation condition must be fully met and that escalation beats clarifying.",
              "<b>Wrong decision labels:</b> asking for an order ID on general questions, calling a plain 'no' a refusal, tagging helpful answers as 'clarify'. Fixed in the prompt.",
              "<b>Missing facts and script:</b> replies left out the seal condition or wrote 'three' for 3, and answered Hinglish in Devanagari. Fixed with explicit wording and script rules.",
              "<b>Intro:</b> the model sometimes skipped 'I'm the AI assistant', so the first reply is checked in code."]:
        f.append(B(t))

    f.append(Paragraph("7. Limitations and next steps", h2))
    for t in ["Retrieval is keyword-based; the expansion rules were written after reading the practice questions, so hidden-set recall will be lower. Next: add dense multilingual embeddings and compare.",
              "Free-tier token limits prevented re-running every evaluation after each fix; results should be re-measured on the full 52 questions and 23 conversations.",
              "Hindi speech is transcribed as Latin text by en-IN; I did not compare hi-IN STT or test other Hindi voices.",
              "Bonus work: query expansion from the product record, date calculator tool, code-level verification and ticket guards, and automatic failure-cause reports."]:
        f.append(B(t))

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                            title="VoiceOps 2026 report")
    doc.build(f)
    from pypdf import PdfReader
    n = len(PdfReader(a.out).pages)
    print(f"Wrote {a.out}: {n} page(s)" + ("  <-- TOO LONG, limit is 4" if n > 4 else "  (limit 4 OK)"))
    for name, s in (("Step 1 cautious", s1c), ("Step 1 guess", s1), ("Step 2", s2), ("Step 3", s3)):
        print(f"  {name:16}: {fmt(s)}")


if __name__ == "__main__":
    main()

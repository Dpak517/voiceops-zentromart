# VoiceOps 2026 - ZentroMart voice support agent

A customer-support agent for the fictional store ZentroMart. It answers from company documents (RAG), holds a
conversation (memory, caller verification, order lookup, tickets, handover) and talks over the phone with Gnani speech models.

**Pipeline:** customer audio -> Gnani STT -> agent (Groq LLM + retrieval + tools) -> Gnani TTS -> spoken reply.

GitHub: https://github.com/Dpak517/voiceops-zentromart

## Setup
1. Python 3.10+.
2. Extract the organizers' dataset ZIP so that `kb/`, `data/`, `practice/` and `tools/` sit **next to** `src/` and `run.py`
   (the dataset is not included in this repo).
3. Install and configure:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   cp .env.example .env            # Windows: copy .env.example .env
   ```
4. Edit `.env`: set `GROQ_API_KEY` (LLM) and `GNANI_API_KEY` (speech). Optional: `GROQ_MODEL`, `GNANI_STT_LANG`, `GNANI_TTS_MODEL`, `GNANI_TTS_VOICE`.

## Run (the one command)
```bash
python run.py --audio-dir <folder_with_customer_audio> --out-dir sample_outputs
```
For every audio file it saves `<name>.json` (what was heard, reply, decision, retrieved chunks and sources),
`<name>_reply.wav` (spoken reply) and a summary `results.csv`. Each file is treated as its own call; add
`--keep-memory` to treat the files, in name order, as one conversation. Add `--simple` for the single-turn Step 2 agent.

Other commands:
```bash
python run.py --text "How long do UPI refunds take?" --speak   # test without audio
python chat.py --debug                                         # talk to the agent in the terminal
python demo.py                                                 # microphone web demo (Gradio)
python -m eval.step1_baseline                                  # Step 1: prompt-only baseline
python -m eval.step2_rag                                       # Step 2: RAG on practice/questions.csv
python -m eval.step3_conversations --ids PC003,PC021           # Step 3: practice conversations
```

## How it works
| Part | File | Idea |
|---|---|---|
| Knowledge base | `src/kb.py` | one chunk per `## [SECTION-ID]`, keeps status / version / date |
| Retrieval | `src/retriever.py`, `src/query.py` | BM25 + query expansion (customer words -> document words) + LLM rewrite for Hindi |
| Records & dates | `src/records.py`, `src/datatools.py` | product / pincode lookup; business-day and calendar-day calculator tools |
| Step 2 agent | `src/rag_agent.py` | answers only from retrieved chunks, says "I don't know" otherwise |
| Step 3 agent | `src/conversation.py` | memory, verification with one retry then lock, tickets via the provided tools |
| Voice | `src/voice.py`, `src/pipeline.py` | Gnani STT / TTS, speech-friendly text, recovery from unclear audio |

Rules followed: Gnani is the only speech service; the LLM is Groq; only the synthetic dataset is used; no policy answers are
hard-coded (they come from the retrieved documents); no keys are committed.

## Known limits
Keyword retrieval (no dense embeddings yet); free-tier token limits; Hindi speech is transcribed by `en-IN` into Latin text.
See `report.pdf` for results and what went wrong.

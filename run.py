"""ONE COMMAND for the organizers:   python run.py --audio-dir <folder_with_audio> --out-dir sample_outputs

For every audio file it saves:  <name>.json (what was heard, the reply, decision, retrieved chunks),
<name>_reply.wav (spoken reply) and one summary results.csv.

Handy extras:
    python run.py --text "How long do refunds take?"          # test the agent without audio
    python run.py --text "..." --speak                        # ...and hear the reply (uses Gnani TTS)
"""
import argparse
import csv
import os
import sys
import time

from src import pipeline
from src.conversation import ConversationAgent
from src.rag_agent import RagAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", help="folder with customer audio files")
    ap.add_argument("--out-dir", default="sample_outputs")
    ap.add_argument("--text", help="run the agent on one text question instead of audio")
    ap.add_argument("--speak", action="store_true", help="with --text: also synthesize speech")
    ap.add_argument("--no-tts", action="store_true", help="skip speech output (text only)")
    ap.add_argument("--stt-lang", default=None, help="Gnani language code, default en-IN")
    ap.add_argument("--simple", action="store_true", help="use the single-turn Step 2 agent (no memory/tools)")
    ap.add_argument("--keep-memory", action="store_true",
                    help="treat the audio files, in name order, as ONE conversation (default: each file is independent)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()

    agent = RagAgent() if a.simple else ConversationAgent()

    if a.text:
        out = pipeline.process_text(agent, a.text)
        print(f"[{out['decision']}] {out['reply']}\nchunks used: {out['context_ids']}")
        if out["error"]:
            print("ERROR:", out["error"])
        if a.speak:
            from src import voice
            os.makedirs(a.out_dir, exist_ok=True)
            print("spoken reply saved:", os.path.join(a.out_dir, "text_reply.wav"),
                  "| language:", voice.synthesize(out["reply"], os.path.join(a.out_dir, "text_reply.wav")))
        return

    if not a.audio_dir or not os.path.isdir(a.audio_dir):
        sys.exit("Give a folder with audio files:  python run.py --audio-dir <folder>")
    files = sorted(f for f in os.listdir(a.audio_dir)
                   if os.path.splitext(f)[1].lower() in pipeline.AUDIO_EXT)
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"No audio files found in {a.audio_dir}")

    rows = []
    for i, name in enumerate(files, 1):
        if hasattr(agent, "reset") and not a.keep_memory:
            agent.reset()                                    # every file is its own call
        r = pipeline.process_audio(agent, os.path.join(a.audio_dir, name), a.out_dir,
                                   speak=not a.no_tts, stt_lang=a.stt_lang)
        rows.append(r)
        print(f"[{i}/{len(files)}] {name}: heard={r['heard'][:60]!r} -> [{r['decision']}] {r['reply'][:70]!r}"
              + (f"  !! {r['error'] or r['stt_error'] or r['tts_error']}" if (r['error'] or r['stt_error'] or r['tts_error']) else ""))
        time.sleep(a.sleep)

    os.makedirs(a.out_dir, exist_ok=True)
    cols = ["file", "heard", "decision", "reply", "audio_out", "tts_language",
            "retrieved_ids", "context_ids", "sources", "stt_error", "error", "tts_error"]
    with open(os.path.join(a.out_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (" ".join(map(str, r[c])) if isinstance(r.get(c), list) else r.get(c, "")) for c in cols})
    print(f"\nDone: {len(rows)} files. Outputs in {a.out_dir}/ (results.csv, <name>.json, <name>_reply.wav)")


if __name__ == "__main__":
    main()

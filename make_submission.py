"""Build <team>_voiceops.zip safely.   python make_submission.py --team myteam
Includes: README.md .env.example requirements.txt run.py demo.py chat.py report.pdf demo.mp4 src/ eval/ sample_outputs/
Never includes: .env, virtual environments, __pycache__, outputs/, or a dataset copy (kb/ data/ practice/ tools/)."""
import argparse
import os
import re
import sys
import zipfile

FILES = ["README.md", ".env.example", "requirements.txt", "run.py", "demo.py", "chat.py", "report.pdf", "demo.mp4"]
DIRS = ["src", "eval", "sample_outputs"]
SKIP = {"__pycache__", ".venv", "venv", ".git", "outputs", "kb", "data", "practice", "tools"}
SECRET = re.compile(r"(gsk_[A-Za-z0-9]{16,})|(API_KEY[ \t]*=[ \t]*[^\s#]+)")


def walk(d):
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            if not f.endswith((".pyc",)) and f != ".env":
                yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team")
    a = ap.parse_args()
    paths = [f for f in FILES if os.path.exists(f)] + [p for d in DIRS if os.path.isdir(d) for p in walk(d)]
    print("MISSING (needed for submission):", [f for f in ("README.md", ".env.example", "requirements.txt", "run.py", "report.pdf", "demo.mp4") if not os.path.exists(f)] or "none")
    if not (os.path.isdir("sample_outputs") and os.listdir("sample_outputs")):
        print("MISSING: sample_outputs/ is empty. Run: python run.py --audio-dir test_audio --out-dir sample_outputs")
    leaked = []
    for p in paths:
        if p.endswith((".py", ".md", ".txt", ".json", ".csv", ".example")):
            try:
                if SECRET.search(open(p, encoding="utf-8", errors="ignore").read()):
                    leaked.append(p)
            except OSError:
                pass
    if leaked:
        sys.exit(f"STOP: possible API key found in {leaked}. Remove it, then run again.")
    out = f"{a.team}_voiceops.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.replace("\\", "/"))
    print(f"Created {out} with {len(paths)} files ({os.path.getsize(out) / 1e6:.1f} MB). No keys found, no .env, no dataset copy.")


if __name__ == "__main__":
    main()

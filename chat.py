"""Talk to the conversation agent in the terminal (no audio, cheap to test).

    python chat.py            # type messages, see decision + reply
    python chat.py --debug    # also show tool calls and state
Commands: 'new' starts a fresh call, 'quit' exits.
"""
import argparse

from src.conversation import ConversationAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    agent = ConversationAgent()
    print("ZentroMart support chat. Type 'new' for a fresh call, 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "new":
            agent.reset()
            print("--- new call ---\n")
            continue
        if not text:
            continue
        out = agent.answer(text)
        print(f"Agent [{out['decision']}]: {out['reply']}\n")
        if a.debug:
            for t in out["tool_calls"]:
                print(f"   tool: {t['tool']}({t['args']}) -> {str(t['result'])[:150]}")
            print(f"   state: {out['state']} | chunks: {out['retrieved_ids']}\n")


if __name__ == "__main__":
    main()

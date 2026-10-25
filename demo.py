"""Web demo for your 2-minute video:  python demo.py   then open the link it prints.
Click the microphone, speak, stop, and the agent answers in text and voice."""
import gradio as gr

from src import pipeline
from src.conversation import ConversationAgent

agent = ConversationAgent()   # remembers the conversation until you tick 'New call'


def talk(audio_path, new_call):
    if new_call:
        agent.reset()
    if not audio_path:
        return "", "", None
    r = pipeline.process_audio(agent, audio_path, "demo_outputs")
    wav = f"demo_outputs/{r['audio_out']}" if r["audio_out"] else None
    return r["heard"], f"[{r['decision']}] {r['reply']}", wav


gr.Interface(
    fn=talk,
    inputs=[gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak to ZentroMart support"),
            gr.Checkbox(label="New call (forget the previous turns)", value=False)],
    outputs=[gr.Textbox(label="What I heard"), gr.Textbox(label="Agent reply"), gr.Audio(label="Agent voice")],
    title="ZentroMart voice support", allow_flagging="never",
).launch()

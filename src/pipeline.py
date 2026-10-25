"""One customer turn end to end:  audio -> Gnani STT -> agent -> Gnani TTS -> audio."""
import json
import os

from . import voice

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"}
FAIL_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."


def process_text(agent, text: str, channel: str = "chat") -> dict:
    """Run the agent on text. Never raises: a failure becomes a polite spoken apology."""
    try:
        out = agent.answer(text, channel=channel)
        out["error"] = ""
        return out
    except Exception as exc:                                    # e.g. rate limit: still reply with speech
        return {"decision": "clarify", "reply": FAIL_REPLY, "sources": [], "retrieved_ids": [],
                "context_ids": [], "search_query": "", "rewrite": "", "tool_calls": [],
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def process_audio(agent, audio_path: str, out_dir: str, speak: bool = True, stt_lang: str = None) -> dict:
    """Handle ONE audio file. Always writes <stem>.json and (if speak) <stem>_reply.wav."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    res = {"file": os.path.basename(audio_path), "heard": "", "stt_error": ""}
    try:
        res["heard"] = voice.transcribe(audio_path, stt_lang).strip()
    except Exception as exc:
        res["stt_error"] = f"{type(exc).__name__}: {exc}"[:300]

    if not res["heard"]:                                        # unclear / empty audio -> recover politely
        out = {"decision": "clarify", "reply": voice.NO_SPEECH_REPLY, "sources": [], "retrieved_ids": [],
               "context_ids": [], "search_query": "", "rewrite": "", "tool_calls": [], "error": ""}
    else:
        out = process_text(agent, res["heard"], channel="voice")
    res.update(out)

    res["audio_out"], res["tts_error"], res["tts_language"] = "", "", ""
    if speak:
        wav = os.path.join(out_dir, f"{stem}_reply.wav")
        try:
            res["tts_language"] = voice.synthesize(res["reply"], wav)
            res["audio_out"] = os.path.basename(wav)
        except Exception as exc:
            res["tts_error"] = f"{type(exc).__name__}: {exc}"[:300]

    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res

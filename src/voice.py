"""Gnani speech layer: STT (ears) and TTS (voice). Only Gnani is used for speech (brief rule)."""
import os
import re
import tempfile
import wave

from dotenv import load_dotenv

load_dotenv()

STT_LANG = os.getenv("GNANI_STT_LANG", "en-IN")        # en-IN also accepts English mixed with Hindi
TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")  # always name the model (brief)
TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Kaveri")
MAX_CLIP_SECONDS = 55                                   # STT REST accepts clips up to 60 s
NO_SPEECH_REPLY = "Sorry, I didn't catch that. Could you say it again?"

_stt = _tts = None


def _stt_client():
    global _stt
    if _stt is None:
        from gnani.stt import GnaniSTTClient            # reads GNANI_API_KEY from the environment
        _stt = GnaniSTTClient()
    return _stt


def _tts_client():
    global _tts
    if _tts is None:
        from gnani.tts import GnaniTTSClient
        _tts = GnaniTTSClient()
    return _tts


def _transcript_of(result) -> str:
    if isinstance(result, dict):
        return str(result.get("transcript") or "")
    return str(result or "")


def _split_wav(path, seconds=MAX_CLIP_SECONDS):
    """Split a long WAV into <=seconds pieces. Returns [path] if short or not a readable WAV."""
    try:
        with wave.open(path, "rb") as w:
            rate, n, params = w.getframerate(), w.getnframes(), w.getparams()
            if n / rate <= seconds:
                return [path], False
            pieces, step = [], int(rate * seconds)
            for i in range(0, n, step):
                w.setpos(i)
                frames = w.readframes(min(step, n - i))
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with wave.open(tmp.name, "wb") as out:
                    out.setparams(params)
                    out.writeframes(frames)
                pieces.append(tmp.name)
            return pieces, True
    except Exception:
        return [path], False


def transcribe(audio_path: str, language_code: str = None) -> str:
    """Speech -> text with Gnani STT. Long WAV files are split and joined."""
    code = language_code or STT_LANG
    pieces, temp = _split_wav(audio_path)
    try:
        return " ".join(_transcript_of(_stt_client().transcribe(p, language_code=code)).strip()
                        for p in pieces).strip()
    finally:
        if temp:
            for p in pieces:
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------- making replies sound right ----------
_HINGLISH = set("""hai hain kya aap aapka aapko aapki mein main hoon ka ki ke ko se par kar karein kijiye
nahi nahin hoga hogi milega milegi kab tak bhi aur mahine din wapas paisa paise ordar liye""".split())


def tts_language(text: str) -> str:
    """Pick the TTS language code from the reply: Devanagari -> hi-IN, Roman Hindi -> hi-en, else en-IN."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    words = re.findall(r"[a-z]+", text.lower())
    return "hi-en" if sum(w in _HINGLISH for w in words) >= 2 else "en-IN"


def speakable(text: str, language: str = "en-IN") -> str:
    """Rewrite a chat reply so it is pleasant to HEAR: no markdown, rupees spelled out,
    long order numbers read in groups of four digits (Gnani text-normalization guide)."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"\[[A-Z]{2,5}-\d+(\.\d+)?\]", "", t)                       # [REF-2] style tags
    word = "रुपये" if language == "hi-IN" else "rupees"
    # 1) order numbers first: long digit runs that are NOT money -> groups of four digits
    t = re.sub(r"(?<![\d,.₹])(?<!₹ )\d{5,}(?![\d,.])(?!\s?(?:rupees|rupaye|rs\b|रुपये|रुपए))",
               lambda m: " ".join(m.group(0)[i:i + 4] for i in range(0, len(m.group(0)), 4)), t)
    # 2) then money: ₹10,000 -> 10000 rupees
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1).replace(',', '')} {word}", t)
    return re.sub(r"\s+", " ", t).strip()


def synthesize(text: str, out_path: str, language: str = None) -> str:
    """Text -> speech WAV with Gnani TTS. Returns the language code used."""
    lang = language or tts_language(text)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _tts_client().synthesize(speakable(text, lang), voice=TTS_VOICE, model=TTS_MODEL,
                             language=lang, output_file=out_path)
    return lang

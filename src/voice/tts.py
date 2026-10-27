"""Local Text-to-Speech (TTS) synthesis module for MuseGuide."""

import io
import logging
import os
import re
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("museguide_tts")

DEFAULT_PIPER_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
DEFAULT_PIPER_JSON_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
)


def clean_text_for_speech(text: str) -> str:
    """Prepare docent answer text for natural speech synthesis.

    Strips inline citation brackets [1], markdown formatting, raw URLs,
    and extraneous symbols so speech sounds natural and conversational.

    Args:
        text: Raw answer markdown text.

    Returns:
        Clean, speech-friendly text.
    """
    if not text:
        return ""

    # 1. Remove bracketed citation markers e.g. [1], [2], [1][2], [1, 2]
    cleaned = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)

    # 2. Remove markdown links [title](url) -> title
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # 3. Remove raw URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)

    # 4. Remove bold/italics markers: ***word***, **word**, *word*
    cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", cleaned)

    # 5. Remove markdown headers (#, ##, ###)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # 6. Normalize dashes, quotes, and whitespace
    cleaned = cleaned.replace("—", ", ").replace("–", ", ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class DocentTTS:
    """Local text-to-speech engine supporting Piper ONNX neural voices with fallback to pyttsx3."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize TTS engine.

        Args:
            model_path: Path to Piper ONNX model file. If None, checks PIPER_MODEL_PATH env or models/piper/en_US-lessac-medium.onnx.
            config_path: Path to Piper JSON config file.
        """
        env_model = os.getenv("PIPER_MODEL_PATH", "models/piper/en_US-lessac-medium.onnx")
        self.model_path = model_path or env_model
        self.config_path = config_path or f"{self.model_path}.json"

        self._piper_voice = None
        self._piper_available = False
        self._engine_type = "uninitialized"

        self._initialize_engine()

    def _initialize_engine(self):
        """Attempt to load Piper ONNX model, otherwise fall back to pyttsx3."""
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            try:
                import piper

                logger.info("Loading Piper ONNX voice from %s...", self.model_path)
                self._piper_voice = piper.PiperVoice.load(self.model_path, config_path=self.config_path)
                self._piper_available = True
                self._engine_type = "piper (en_US-lessac-medium)"
                logger.info("Piper neural TTS successfully initialized.")
                return
            except Exception as e:
                logger.warning("Failed to load Piper voice (%s). Falling back to pyttsx3.", e)

        # Fallback to local pyttsx3 (uses native OS SAPI5 on Windows / NSSpeechSynthesizer on macOS)
        try:
            import pyttsx3

            # Verify pyttsx3 can be initialized
            engine = pyttsx3.init()
            engine.stop()
            self._engine_type = "pyttsx3 (local OS voice)"
            logger.info("Local OS speech engine (pyttsx3) available as fallback.")
        except Exception as e:
            logger.warning("pyttsx3 initialization failed (%s). TTS will be unavailable until a model is provided.", e)
            self._engine_type = "unavailable"

    @property
    def engine_name(self) -> str:
        """Name of active speech generation engine."""
        return self._engine_type

    def is_available(self) -> bool:
        """Check if any TTS engine is ready to synthesize speech."""
        return self._engine_type != "unavailable"

    def synthesize_wav(self, text: str) -> bytes:
        """Synthesize spoken audio as WAV bytes.

        Args:
            text: Raw or clean docent answer text.

        Returns:
            Raw bytes of standard WAV audio.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If audio synthesis fails.
        """
        clean_text = clean_text_for_speech(text)
        if not clean_text or len(clean_text) < 3:
            raise ValueError("Text is empty or too short to synthesize speech.")

        # 1. Primary: High-fidelity Piper neural ONNX synthesis
        if self._piper_available and self._piper_voice is not None:
            try:
                chunks = list(self._piper_voice.synthesize(clean_text))
                if not chunks:
                    raise RuntimeError("Piper synthesis returned empty audio chunks.")

                buf = io.BytesIO()
                first = chunks[0]
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(first.sample_channels)
                    wf.setsampwidth(first.sample_width)
                    wf.setframerate(first.sample_rate)
                    for c in chunks:
                        wf.writeframes(c.audio_int16_bytes)

                audio_data = buf.getvalue()
                logger.info("Piper synthesized %d bytes of speech for %d characters.", len(audio_data), len(clean_text))
                return audio_data
            except Exception as e:
                logger.warning("Piper synthesis error (%s); falling back to pyttsx3.", e)

        # 2. Secondary Fallback: pyttsx3 local system speech
        try:
            import pyttsx3

            engine = pyttsx3.init()
            # Set speech rate and volume for docent tone
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 0.95)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                engine.save_to_file(clean_text, tmp_path)
                engine.runAndWait()

                with open(tmp_path, "rb") as f:
                    audio_data = f.read()

                if not audio_data:
                    raise RuntimeError("pyttsx3 generated empty audio file.")

                logger.info("pyttsx3 synthesized %d bytes of speech.", len(audio_data))
                return audio_data
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.error("All local TTS engines failed to synthesize audio: %s", e)
            raise RuntimeError(f"Local speech synthesis failed: {e}") from e


def download_piper_model(target_dir: str = "models/piper") -> Tuple[str, str]:
    """Download the official Piper en_US-lessac-medium ONNX voice model and config.

    Args:
        target_dir: Directory where models are saved.

    Returns:
        Tuple of (onnx_path, json_path).
    """
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)

    onnx_path = dest / "en_US-lessac-medium.onnx"
    json_path = dest / "en_US-lessac-medium.onnx.json"

    if not json_path.exists():
        logger.info("Downloading Piper config: %s", json_path)
        urllib.request.urlretrieve(DEFAULT_PIPER_JSON_URL, str(json_path))

    if not onnx_path.exists():
        logger.info("Downloading Piper ONNX voice binary (~60 MB): %s", onnx_path)
        urllib.request.urlretrieve(DEFAULT_PIPER_ONNX_URL, str(onnx_path))

    return str(onnx_path), str(json_path)

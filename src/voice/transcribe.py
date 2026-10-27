"""Groq Whisper Speech-to-Text Transcription Module for MuseGuide."""

import io
import logging
import os
from typing import Optional, Union

logger = logging.getLogger("museguide_voice")


class VoiceTranscriber:
    """Transcribes user voice queries using Groq's official Whisper API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the speech-to-text transcriber.

        Args:
            api_key: Optional Groq API key. If None, reads GROQ_API_KEY from environment.
            model: Optional Whisper model. If None, reads GROQ_WHISPER_MODEL or defaults to 'whisper-large-v3-turbo'.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

    def transcribe(
        self,
        audio_data: Union[bytes, io.BytesIO, any],
        filename: str = "query.wav",
        prompt: Optional[str] = None,
    ) -> str:
        """Transcribe audio bytes to natural language text.

        Args:
            audio_data: Raw audio bytes or file-like buffer (e.g. from st.audio_input).
            filename: Virtual filename with audio extension (e.g. .wav).
            prompt: Optional context prompt to guide transcription (e.g., museum terminology).

        Returns:
            Transcribed text query.

        Raises:
            ValueError: If audio is empty or API key is missing.
            RuntimeError: If Groq Whisper transcription fails.
        """
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please configure your key in .env or choose typing instead."
            )

        # Extract bytes from various buffer types
        if hasattr(audio_data, "read"):
            # If it's a Streamlit UploadedFile or BytesIO, read bytes
            audio_bytes = audio_data.read()
            # Reset pointer if possible
            if hasattr(audio_data, "seek"):
                audio_data.seek(0)
        elif isinstance(audio_data, (bytes, bytearray)):
            audio_bytes = bytes(audio_data)
        elif isinstance(audio_data, io.BytesIO):
            audio_bytes = audio_data.getvalue()
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        if not audio_bytes or len(audio_bytes) < 120:
            raise ValueError("Audio recording is empty or too short. Please speak clearly into your microphone.")

        try:
            from groq import Groq
        except ImportError as e:
            raise RuntimeError("The 'groq' package is required for voice transcription. Install via `pip install groq`.") from e

        try:
            client = Groq(api_key=self.api_key, timeout=30.0)
            logger.info("Sending %d bytes of audio to Groq Whisper model: %s", len(audio_bytes), self.model)

            extra_kwargs = {}
            if prompt:
                extra_kwargs["prompt"] = prompt

            transcription = client.audio.transcriptions.create(
                file=(filename, audio_bytes, "audio/wav"),
                model=self.model,
                response_format="json",
                temperature=0.0,
                **extra_kwargs,
            )

            result_text = getattr(transcription, "text", "") or ""
            clean_text = result_text.strip()
            if not clean_text:
                raise ValueError("No speech could be recognized from the recording. Please try speaking again.")

            logger.info("Transcription succeeded: %s", clean_text)
            return clean_text

        except Exception as e:
            err_msg = str(e)
            logger.error("Groq Whisper transcription failed: %s", err_msg)
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                raise RuntimeError("Groq Whisper rate limit reached. Please wait a moment or type your question.") from e
            if "invalid_api_key" in err_msg.lower() or "401" in err_msg:
                raise ValueError("Invalid GROQ_API_KEY provided. Please check your .env configuration.") from e
            raise RuntimeError(f"Voice transcription failed: {err_msg}") from e

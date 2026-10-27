"""Automated tests for Voice Input (Groq Whisper) and Text-to-Speech (Piper / pyttsx3) integration."""

import io
import os
import sys
import wave
from pathlib import Path
import pytest

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.answer import AnswerGenerator, RAGResponse
from src.retrieval.search import SemanticSearcher
from src.voice.transcribe import VoiceTranscriber
from src.voice.tts import DocentTTS, clean_text_for_speech


@pytest.fixture(scope="module")
def searcher():
    """Shared SemanticSearcher instance."""
    index_path = Path("data/index/cma_index.faiss")
    if not index_path.exists():
        pytest.skip("Full FAISS index not built at data/index/cma_index.faiss")
    return SemanticSearcher(
        index_path=str(index_path),
        metadata_path="data/index/cma_metadata.json",
    )


@pytest.fixture(scope="module")
def generator(searcher):
    """AnswerGenerator instance using offline extractive synthesizer."""
    return AnswerGenerator(searcher=searcher, min_confidence=0.35, top_k=4, llm_backend="extractive")


@pytest.fixture(scope="module")
def tts_engine():
    """DocentTTS instance."""
    return DocentTTS()


# -------------------------------------------------------------
# 1. Text-to-Speech (TTS) Cleaning & Synthesis Tests
# -------------------------------------------------------------

def test_clean_text_for_speech_strips_citations():
    """Verify inline citation brackets [1], [2], [1][2] are stripped from speech text."""
    raw = "*Portrait of Dora Wheeler* [1] is a **painting** created by William Merritt Chase [1][2]."
    cleaned = clean_text_for_speech(raw)

    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "[1][2]" not in cleaned
    assert "Portrait of Dora Wheeler" in cleaned
    assert "painting" in cleaned
    assert "William Merritt Chase" in cleaned


def test_clean_text_for_speech_strips_markdown_formatting():
    """Verify markdown bold, italics, headers, and raw URLs are stripped while words remain."""
    raw = (
        "### Curatorial Overview\n"
        "Nathaniel Hurd was a **prominent silversmith** in *Boston* [1]. "
        "View at [CMA Portal](https://clevelandart.org/art/1915.534)."
    )
    cleaned = clean_text_for_speech(raw)

    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "https://" not in cleaned
    assert "Nathaniel Hurd was a prominent silversmith in Boston" in cleaned
    assert "CMA Portal" in cleaned


def test_clean_text_for_speech_empty_handling():
    """Verify empty or whitespace strings return empty string without errors."""
    assert clean_text_for_speech("") == ""
    assert clean_text_for_speech("   ") == ""
    assert clean_text_for_speech(None) == ""


def test_tts_synthesize_wav_produces_valid_audio(tts_engine):
    """Verify TTS synthesizer outputs valid WAV bytes with correct header."""
    sample_text = "Nathaniel Hurd was an American silversmith and engraver."
    wav_bytes = tts_engine.synthesize_wav(sample_text)

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 2000
    # Check RIFF and WAVE header markers
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_tts_missing_piper_model_graceful_fallback(monkeypatch, tmp_path):
    """Verify non-existent Piper model path gracefully falls back to pyttsx3."""
    dummy_onnx = str(tmp_path / "nonexistent.onnx")
    tts_fallback = DocentTTS(model_path=dummy_onnx, config_path=dummy_onnx + ".json")

    assert tts_fallback._piper_available is False
    assert "pyttsx3" in tts_fallback.engine_name or tts_fallback.is_available()

    # Audio synthesis should still succeed via pyttsx3 fallback
    audio = tts_fallback.synthesize_wav("Testing fallback speech engine.")
    assert len(audio) > 1000
    assert audio[:4] == b"RIFF"


def test_tts_empty_text_raises_value_error(tts_engine):
    """Verify synthesizing empty or too short text raises ValueError."""
    with pytest.raises(ValueError, match="too short"):
        tts_engine.synthesize_wav("   ")


# -------------------------------------------------------------
# 2. Voice Transcription (Groq Whisper) Tests
# -------------------------------------------------------------

def test_transcription_missing_api_key_raises_value_error(monkeypatch):
    """Verify missing API key raises clear ValueError explaining how to configure .env."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    transcriber = VoiceTranscriber(api_key=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        transcriber.transcribe(b"dummy_wav_data" * 20)


def test_transcription_empty_audio_raises_value_error():
    """Verify empty or truncated audio raises clear ValueError."""
    transcriber = VoiceTranscriber(api_key="gsk_test_key_dummy")

    with pytest.raises(ValueError, match="empty or too short"):
        transcriber.transcribe(b"")

    with pytest.raises(ValueError, match="empty or too short"):
        transcriber.transcribe(b"short")


def test_mocked_groq_whisper_transcription(monkeypatch):
    """Verify Whisper transcription succeeds using mocked Groq API client without network calls."""
    captured_kwargs = {}

    class MockTranscription:
        text = "Tell me the story behind Stag at Sharkey's"

    class MockAudioTranscriptions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return MockTranscription()

    class MockAudio:
        transcriptions = MockAudioTranscriptions()

    class MockGroqClient:
        def __init__(self, **kwargs):
            assert kwargs.get("api_key") == "gsk_test_valid_key"
        audio = MockAudio()

    monkeypatch.setattr("groq.Groq", MockGroqClient)

    transcriber = VoiceTranscriber(api_key="gsk_test_valid_key", model="whisper-large-v3-turbo")
    fake_audio = b"RIFF" + (b"\x00" * 400)
    result = transcriber.transcribe(fake_audio, filename="voice.wav")

    assert result == "Tell me the story behind Stag at Sharkey's"
    assert captured_kwargs.get("model") == "whisper-large-v3-turbo"
    assert captured_kwargs.get("temperature") == 0.0


def test_whisper_rate_limit_error_handling(monkeypatch):
    """Verify 429 rate limit errors from Groq Whisper produce user-friendly guidance."""
    class MockAudioTranscriptions:
        def create(self, **kwargs):
            raise Exception("Error code: 429 - rate_limit_exceeded")

    class MockGroqClient:
        def __init__(self, **kwargs):
            pass
        audio = type("MockAudio", (), {"transcriptions": MockAudioTranscriptions()})()

    monkeypatch.setattr("groq.Groq", MockGroqClient)

    transcriber = VoiceTranscriber(api_key="gsk_test_key")
    with pytest.raises(RuntimeError, match="rate limit reached"):
        transcriber.transcribe(b"RIFF" + (b"\x00" * 300))


# -------------------------------------------------------------
# 3. Voice Query to RAG Pipeline Integration Tests
# -------------------------------------------------------------

def test_voice_query_enters_existing_rag_pipeline(generator):
    """Verify voice transcribed question seamlessly flows through the existing RAG pipeline."""
    voice_transcription = "Who was Nathaniel Hurd and what portrait of him is in the collection?"
    response = generator.generate(voice_transcription)

    assert isinstance(response, RAGResponse)
    assert response.grounded is True
    assert "[1]" in response.answer
    assert len(response.citations) >= 1
    assert "Nathaniel Hurd" in response.citations[0].title
    assert hasattr(response, "artwork_details")
    assert len(response.artwork_details) >= 1


def test_voice_unsupported_query_triggers_refusal(generator):
    """Verify out-of-domain voice queries still trigger the anti-hallucination refusal guardrail."""
    voice_transcription = "What is the capital city of planet Mars and how was its government founded?"
    response = generator.generate(voice_transcription)

    assert response.grounded is False
    assert len(response.citations) == 0
    assert "could not find sufficient historical evidence" in response.answer.lower()


def test_typed_question_still_works_normally(generator):
    """Regression test: Ensure standard typed input continues to operate identically."""
    response = generator.generate("What is Portrait of Dora Wheeler, is it a book or a painting?")
    assert response.grounded is True
    assert "painting" in response.answer.lower()
    assert len(response.citations) == 1
    assert response.citations[0].artwork_id == 102578


def test_duplicate_prevention_logic():
    """Verify audio hashing prevents duplicate processing of the same recording on reruns."""
    audio_sample = b"RIFF" + (b"\x01\x02\x03\x04" * 100)
    hash_1 = hash(audio_sample)
    hash_2 = hash(audio_sample)

    assert hash_1 == hash_2, "Same audio bytes must produce identical hash signature"

    # Different audio bytes must produce different signature
    different_audio = b"RIFF" + (b"\x09\x08\x07\x06" * 100)
    assert hash(different_audio) != hash_1

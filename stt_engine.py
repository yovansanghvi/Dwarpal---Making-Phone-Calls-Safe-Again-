import audioop
import os
import sys
import tempfile
import threading
import warnings
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from silero_vad import load_silero_vad
from transformers import AutoModel

warnings.filterwarnings("ignore", category=UserWarning)


class SharedSTTEngine:
    """Stateless GPU inference engine shared safely across concurrent calls/tracks."""

    def __init__(self, model_id: str = "bodhan-ai/indic-transcribe-flex", device: str = "cuda"):
        print("[STT] Loading Silero VAD...")
        self.vad_model = load_silero_vad()
        self.device = device
        self.sample_rate = 16000
        self._gpu_lock = threading.Lock()
        self._vad_lock = threading.Lock()

        print(f"[STT] Loading {model_id} in FP16 on {device}...")
        model_dir = snapshot_download(repo_id=model_id, local_files_only=True)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.asr_model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self.asr_model.eval()

        from indic_transcribe import IndicTranscribe
        from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
        from tokenization_indic_canary import IndicCanaryTokenizer

        feature_extractor = IndicCanaryFeatureExtractor.from_pretrained(model_dir)
        tokenizer = IndicCanaryTokenizer.from_pretrained(model_dir)

        # Keep internal STFT Hann window / filterbanks aligned on target device
        if hasattr(feature_extractor, "to"):
            feature_extractor = feature_extractor.to(device)
        for attr_name in dir(feature_extractor):
            attr_val = getattr(feature_extractor, attr_name, None)
            if isinstance(attr_val, torch.Tensor):
                setattr(feature_extractor, attr_name, attr_val.to(device))

        self.asr = IndicTranscribe(
            model=self.asr_model,
            fe=feature_extractor,
            tokenizer=tokenizer,
            device=device,
        )

    def create_session(self, call_id: str, track: str = "inbound") -> "CallSession":
        return CallSession(engine=self, call_id=call_id, track=track)

    def get_speech_prob(self, window: np.ndarray) -> float:
        tensor_window = torch.from_numpy(window)
        with self._vad_lock:
            return self.vad_model(tensor_window, self.sample_rate).item()

    def transcribe_audio(self, audio_pcm: np.ndarray, mode: str = "mixed") -> dict | None:
        if len(audio_pcm) < int(0.35 * self.sample_rate):
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_pcm, self.sample_rate)

            with self._gpu_lock, torch.no_grad():
                try:
                    result = self.asr.transcribe(
                        tmp_path,
                        mode=mode,
                        allowed_langs=["hi", "gu", "en"],
                        return_lid=True,
                    )
                except TypeError:
                    result = self.asr.transcribe(
                        tmp_path,
                        mode=mode,
                        return_lid=True,
                    )

            if isinstance(result, tuple):
                text, lid_info = result
                detected_lang = (
                    lid_info.get("lang", "AUTO").upper()
                    if isinstance(lid_info, dict)
                    else str(lid_info).upper()
                )
            else:
                text = str(result)
                detected_lang = "AUTO"

            text = text.strip()
            if not text:
                return None

            return {
                "lang": detected_lang,
                "text": text,
                "formatted": f"[{detected_lang}] {text}",
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


class CallSession:
    """Isolated per-call / per-track audio buffer, VAD state, and transcript history."""

    def __init__(self, engine: SharedSTTEngine, call_id: str, track: str = "inbound"):
        self.engine = engine
        self.call_id = call_id
        self.track = track
        self.sample_rate = engine.sample_rate

        # Isolated audio state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_samples = 0

        # VAD segmentation thresholds
        self.vad_window_size = 512  # 32ms at 16kHz
        self.silence_threshold_samples = int(0.65 * self.sample_rate)     # Normal sentence pause
        self.soft_split_samples = int(7.5 * self.sample_rate)            # Look for micro-pause after 7.5s
        self.micro_pause_samples = int(0.16 * self.sample_rate)          # 160ms breath pause
        self.max_speech_samples = int(10.0 * self.sample_rate)           # Hard ceiling during non-stop speech

        # Rolling history ready for the upcoming Scam Detection Engine
        self.transcript_history: list[dict] = []

    def decode_twilio_chunk(self, ulaw_bytes: bytes) -> np.ndarray:
        pcm_16bit_8k = audioop.ulaw2lin(ulaw_bytes, 2)
        audio_8k = np.frombuffer(pcm_16bit_8k, dtype=np.int16).astype(np.float32) / 32768.0
        return resample_poly(audio_8k, up=2, down=1).astype(np.float32)

    def ingest_chunk(self, ulaw_bytes: bytes) -> list[np.ndarray]:
        """Fast CPU-only VAD step (<1ms). Returns ready speech segments without blocking."""
        chunk_16k = self.decode_twilio_chunk(ulaw_bytes)
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk_16k))
        ready_segments: list[np.ndarray] = []

        while len(self.audio_buffer) >= self.vad_window_size:
            window = self.audio_buffer[: self.vad_window_size]
            self.audio_buffer = self.audio_buffer[self.vad_window_size :]

            speech_prob = self.engine.get_speech_prob(window)

            if speech_prob > 0.5:
                self.is_speaking = True
                self.silence_samples = 0
                self.speech_buffer = np.concatenate((self.speech_buffer, window))
            elif self.is_speaking:
                self.silence_samples += self.vad_window_size
                self.speech_buffer = np.concatenate((self.speech_buffer, window))

            # Unconditional boundary check: triggers even during continuous non-stop speech
            if self.is_speaking:
                buf_len = len(self.speech_buffer)
                normal_pause = self.silence_samples >= self.silence_threshold_samples
                smart_soft_split = (
                    buf_len >= self.soft_split_samples
                    and self.silence_samples >= self.micro_pause_samples
                )
                hard_ceiling_hit = buf_len >= self.max_speech_samples

                if normal_pause or smart_soft_split or hard_ceiling_hit:
                    segment = self.flush_buffer()
                    if segment is not None:
                        ready_segments.append(segment)

        return ready_segments

    def flush_buffer(self) -> np.ndarray | None:
        if len(self.speech_buffer) < int(0.35 * self.sample_rate):
            self.reset_state()
            return None
        segment = self.speech_buffer.copy()
        self.reset_state()
        return segment

    def record_turn(self, turn_data: dict):
        turn_entry = {
            "track": self.track,
            "lang": turn_data["lang"],
            "text": turn_data["text"],
        }
        self.transcript_history.append(turn_entry)

    def reset_state(self):
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_samples = 0
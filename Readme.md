# Dwarpal — Making Phone calls safe again!

**Dwarpal** is an offline-capable, real-time voice-call scam detection system that listens to incoming call audio, transcribes multilingual speech on the fly, and flags social-engineering and vishing tactics such as fake authority threats, false urgency, secrecy isolation, OTP extraction, and coerced money transfers before the victim falls into the trap.

---
# Demo Link - https://www.youtube.com/watch?v=FWiTNgiJE4A

---
## Key Features

- **Caller-Only Voice Isolation:** Captures and analyzes only the incoming caller's voice stream while ignoring the recipient's own microphone speech.
- **Multilingual & Code-Switched Transcription:** Supports real-time speech-to-text across **Hindi**, **Gujarati**, and **English**, including natural code-switched conversations.
- **3-Tier Low-Latency Detection Pipeline:**
  1. **Listen:** Streams and transcribes incoming caller speech live without blocking the call.
  2. **Quick Check:** Runs fast deterministic behavioral rules alongside lightweight sentence-level classification (`mDeBERTa-v3`) to keep benign calls `SAFE` and escalate suspicious utterances to `SUSPICIOUS`.
  3. **Deep Check:** Executes asynchronous multi-turn context verification (`Qwen2.5-1.5B`) in the background to either clear false positives back to `SAFE` or latch the call state into `CRITICAL_SCAM`.
- **Real-Time Risk State Tracking:** Continuously updates and displays call risk states (`SAFE` → `SUSPICIOUS` → `CRITICAL_SCAM`) as each turn is spoken.
- **100% Local & Privacy-First Execution:** Runs all speech recognition and scam detection models locally on-device with zero cloud audio uploads and zero third-party inference API calls.

---

## AI Models & APIs Used

### AI Models
- **`bodhan-ai/indic-transcribe-flex` (Indic-Transcribe-Flex):** Multilingual automatic speech recognition (ASR) model used to transcribe live incoming caller audio across Hindi, Gujarati, and English.
- **`mDeBERTa-v3`:** Multilingual transformer classifier paired with behavioral pattern rules for fast, per-turn risk scoring in the Quick Check stage.
- **`Qwen/Qwen2.5-1.5B` (Qwen2.5-1.5B):** Lightweight local large language model used in the asynchronous Deep Check stage to analyze multi-turn conversational context and confirm high-risk scam patterns.

### APIs & Runtime Interfaces
- **External Cloud / Paid APIs:** None. All model inference runs locally on-device without sending call audio or transcripts to external cloud APIs.
- **Local Runtime & Audio APIs:** Uses local Python runtime bindings, Hugging Face local model loading (`transformers`), PyTorch (`torch`), and system/virtual audio capture interfaces for live call audio streaming.

---

## How to Run the Project

### 1. Prerequisites
- Python 3.10+
- CUDA-compatible GPU recommended (CPU execution also supported with higher latency)
- Local microphone / virtual audio routing configured to capture incoming caller audio

### 2. Install Required Libraries
Install the core dependencies directly via `pip`:

```bash
pip install torch transformers accelerate sounddevice numpy scipy sentencepiece protobuf
```

*(If running `Indic-Transcribe-Flex` via NVIDIA NeMo instead of Hugging Face `transformers`, also install `nemo_toolkit[asr]`.)*

### 3. Run Dwarpal
Clone or download this repository, open a terminal inside the project folder, and launch the main script:

```bash
python main.py
```



1. Wait for the local models (`Indic-Transcribe-Flex`, `mDeBERTa-v3`, and `Qwen2.5-1.5B`) to load into memory.
2. Start or play an incoming call through your configured audio input channel.
3. Observe live transcriptions and real-time risk transitions (`SAFE`, `SUSPICIOUS`, `CRITICAL_SCAM`) directly in the console/monitor view.

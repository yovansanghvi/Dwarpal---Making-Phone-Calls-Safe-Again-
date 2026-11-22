import json
import re
import threading
from dataclasses import dataclass, field
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def normalize_to_roman(text: str) -> str:
    """Converts Gujarati & Devanagari characters into phonetic Roman script for unified matching."""
    out = transliterate(text, sanscript.GUJARATI, sanscript.ITRANS)
    out = transliterate(out, sanscript.DEVANAGARI, sanscript.ITRANS)
    out = out.lower()
    out = re.sub(r"[^a-z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


BEHAVIORAL_SIGNALS: dict[str, list[str]] = {
    "secret_extraction": [
        r"\botp\b", r"\bcvv\b", r"\bpin\b", r"\bpassword\b", r"\bcode\b",
        r"\botipi\b", r"\bverify\b", r"\bverification\b", r"\bshare\b",
        r"\bscreen\s*share\b", r"\banydesk\b", r"\bteamviewer\b", r"\bquicksupport\b",
        r"\brustdesk\b", r"\bapk\b", r"\bdownload\b", r"\binstall\b",
    ],
    "fear_loss_threat": [
        r"\bblock\b", r"\bsuspend\b", r"\bfreeze\b", r"\bband\b", r"\bdeactivate\b",
        r"\bdisconnect\b", r"\bpaisa\b", r"\bpaisaa\b", r"\bkat\b", r"\bkata\b",
        r"\bkapashe\b", r"\bdeduct\b", r"\bdebit\b", r"\bcharge\b", r"\bpenalty\b",
        r"\bfine\b", r"\barrest\b", r"\bwarrant\b", r"\bfir\b", r"\bcase\b",
        r"\billegal\b", r"\bdrugs\b", r"\bnarcotics\b", r"\bexpire\b", r"\bkyc\b",
    ],
    "artificial_urgency": [
        r"\bjaldi\b", r"\bjaldii\b", r"\babhi\b", r"\babhii\b", r"\bturat\b",
        r"\bimmediately\b", r"\burgent\b", r"\bright\s*now\b", r"\bwithin\b",
        r"\bminute\b", r"\bminutes\b", r"\btoday\b", r"\baaj\b", r"\baaje\b",
        r"\bfast\b", r"\bquick\b", r"\bquickly\b", r"\bwarna\b", r"\bvarna\b",
        r"\bnahitar\b", r"\bwapas\b", r"\bvaapas\b",
    ],
    "authority_impersonation": [
        r"\bbank\b", r"\brbi\b", r"\bsbi\b", r"\bhdfc\b", r"\bicici\b", r"\baxis\b",
        r"\btrai\b", r"\bcbi\b", r"\bpolice\b", r"\bcyber\b", r"\bcustoms\b",
        r"\bcourier\b", r"\bparcel\b", r"\bfedex\b", r"\bdhl\b", r"\bdepartment\b",
        r"\bofficer\b", r"\bhead\s*office\b", r"\bcustomer\s*care\b", r"\bmanager\b",
        r"\bvideo\s*call\b",
    ],
    "payment_trap": [
        r"\bupi\b", r"\bqr\b", r"\bscan\b", r"\bcollect\b", r"\brefund\b",
        r"\bcashback\b", r"\btransfer\b", r"\bpay\b", r"\bpayment\b", r"\bclaim\b",
        r"\blottery\b", r"\bprize\b", r"\btask\b",
    ],
    "secrecy_isolation": [
        r"\bkisi\s*ko\b", r"\bkone\s*pan\b", r"\bdont\s*tell\b", r"\bdo\s*not\s*tell\b",
        r"\bcut\s*mat\b", r"\bhold\s*pe\b", r"\bdisconnect\s*mat\b", r"\bconfidential\b",
        r"\bsecret\b", r"\bpersonal\b",
    ],
}


@dataclass
class CallRiskState:
    call_id: str
    triggered_dimensions: dict[str, set[str]] = field(default_factory=dict)
    turn_scores: list[float] = field(default_factory=list)
    peak_score: float = 0.0
    alert_triggered: bool = False
    llm_in_flight: bool = False
    confirmed_scam: bool = False
    cooldown_turns_remaining: int = 0
    last_llm_verdict: dict | None = None


class HybridScamDetector:
    """
    Hierarchical 3-Tier Real-Time Scam Detector:
      - Fast Path (Every Turn): Tier 1 Behavioral State + Tier 2 Multilingual XNLI (~35ms CPU)
      - Slow Deliberative Path (Async on Threshold): Tier 3 Qwen2.5-1.5B reads past 10 turns
    """

    def __init__(
        self,
        nli_model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        nli_device: str = "cpu",
        llm_device: str = "cuda",
        shared_gpu_lock: threading.Lock | None = None,
    ):
        self.nli_device = nli_device
        self.llm_device = llm_device
        self._nli_lock = threading.Lock()
        self._llm_lock = shared_gpu_lock or threading.Lock()
        self.call_states: dict[str, CallRiskState] = {}

        print(f"[SCAM] Loading Tier 2 Semantic Classifier ({nli_model_id}) on {nli_device}...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
        nli_dtype = torch.float16 if nli_device == "cuda" else torch.float32
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_id,
            torch_dtype=nli_dtype,
        ).to(nli_device)
        self.nli_model.eval()

        self.entailment_idx = 0
        for idx, label in self.nli_model.config.id2label.items():
            if "entail" in label.lower():
                self.entailment_idx = int(idx)
                break

        self.hypotheses = {
            "coercion_ask": (
                "The caller asks for an OTP, PIN, password, money transfer, or verification code."
            ),
            "threat_impersonation": (
                "The caller threatens arrest, police, CBI, FIR, customs, or bank account blocking."
            ),
            "benign_normal": (
                "This is a normal everyday chat between friends, classmates, or family about routine topics."
            ),
        }

        print(f"[SCAM] Loading Tier 3 Contextual Reasoner ({llm_model_id}) in 4-bit on {llm_device}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        if llm_device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                quantization_config=bnb_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=torch.float32,
            ).to(llm_device)
        self.llm_model.eval()

    def get_or_create_state(self, call_id: str) -> CallRiskState:
        if call_id not in self.call_states:
            self.call_states[call_id] = CallRiskState(call_id=call_id)
        return self.call_states[call_id]

    def clear_call(self, call_id: str):
        self.call_states.pop(call_id, None)

    def _score_behavioral_state(self, state: CallRiskState, roman_text: str) -> tuple[float, list[str], list[str]]:
        for dimension, patterns in BEHAVIORAL_SIGNALS.items():
            for pat in patterns:
                match = re.search(pat, roman_text)
                if match:
                    state.triggered_dimensions.setdefault(dimension, set()).add(match.group(0))

        active_dims = list(state.triggered_dimensions.keys())
        core_dims = [d for d in active_dims if d != "artificial_urgency"]

        if not core_dims:
            return (0.10 if "artificial_urgency" in active_dims else 0.0), active_dims, core_dims

        weights = {
            "secret_extraction": 0.42,
            "payment_trap": 0.38,
            "fear_loss_threat": 0.30,
            "authority_impersonation": 0.25,
            "secrecy_isolation": 0.25,
            "artificial_urgency": 0.20,
        }
        raw = sum(weights.get(d, 0.15) for d in active_dims)

        if "secret_extraction" in active_dims and (
            "fear_loss_threat" in active_dims or "artificial_urgency" in active_dims
        ):
            raw += 0.25
        if "authority_impersonation" in active_dims and "fear_loss_threat" in active_dims:
            raw += 0.25
        if len(core_dims) >= 3:
            raw += 0.15

        return min(raw, 1.0), active_dims, core_dims

    def _score_semantic_intent(self, rolling_context: str) -> dict[str, float]:
        premises = [rolling_context] * len(self.hypotheses)
        hyp_keys = list(self.hypotheses.keys())
        hyp_texts = [self.hypotheses[k] for k in hyp_keys]

        inputs = self.nli_tokenizer(
            premises,
            hyp_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.nli_device)

        with self._nli_lock, torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, self.entailment_idx].cpu().tolist()

        return dict(zip(hyp_keys, probs))

    def analyze_fast_tiers(self, call_id: str, transcript_history: list[dict]) -> dict:
        """
        TIER 1 + TIER 2 (Non-blocking Fast Loop):
        Runs on every turn even while Tier 3 LLM is deliberating in the background.
        """
        if not transcript_history:
            return {"risk_score": 0.0, "level": "SAFE", "trigger_tier3": False}

        state = self.get_or_create_state(call_id)
        latest_text = transcript_history[-1]["text"]
        roman_latest = normalize_to_roman(latest_text)

        # Tier 1: Stateful Behavioral Score
        behavioral_score, active_dims, core_dims = self._score_behavioral_state(state, roman_latest)

        # Tier 2: Rolling 5-Turn Semantic Context Score
        recent_turns = [t["text"] for t in transcript_history[-5:]]
        rolling_window_text = " | ".join(recent_turns)
        nli_scores = self._score_semantic_intent(rolling_window_text)

        malicious_semantic = max(
            nli_scores["coercion_ask"],
            nli_scores["threat_impersonation"],
        )
        semantic_risk = malicious_semantic * max(0.15, 1.0 - 0.85 * nli_scores["benign_normal"])

        if not core_dims:
            semantic_risk = min(semantic_risk, 0.35)

        fused_score = round(min(1.0, 0.55 * behavioral_score + 0.45 * semantic_risk), 3)

        if state.confirmed_scam:
            fused_score = round(max(fused_score, 0.88), 3)
        elif state.turn_scores and state.cooldown_turns_remaining == 0:
            prev_score = state.turn_scores[-1]
            fused_score = round(max(fused_score, prev_score * 0.85), 3)

        if state.cooldown_turns_remaining > 0:
            state.cooldown_turns_remaining -= 1

        state.turn_scores.append(fused_score)
        state.peak_score = max(state.peak_score, fused_score)

        if fused_score >= 0.72:
            level = "CRITICAL_SCAM"
        elif fused_score >= 0.48:
            level = "SUSPICIOUS"
        else:
            level = "SAFE"

        trigger_tier3 = (
            fused_score >= 0.48
            and not state.llm_in_flight
            and not state.confirmed_scam
            and state.cooldown_turns_remaining == 0
        )

        should_alert = fused_score >= 0.82 and not state.alert_triggered
        if should_alert:
            state.alert_triggered = True

        return {
            "risk_score": fused_score,
            "level": level,
            "behavioral_score": round(behavioral_score, 3),
            "semantic_risk": round(semantic_risk, 3),
            "active_tactics": active_dims,
            "trigger_tier3": trigger_tier3,
            "llm_in_flight": state.llm_in_flight,
            "alert": should_alert,
            "warning": state.last_llm_verdict["warning"] if state.last_llm_verdict else None,
        }

    def run_tier3_deliberation(self, call_id: str, past_10_turns: list[dict], active_dims: list[str]) -> dict | None:
        """
        TIER 3 (Slow Deliberative Loop):
        Reads the full past 10 turns asynchronously while Tier 1+2 continue monitoring live audio.
        """
        state = self.call_states.get(call_id)
        if not state:
            return None

        formatted_turns = "\n".join(
            f"Turn {i + 1} [{t['lang']}]: {t['text']}" for i, t in enumerate(past_10_turns)
        )

        system_prompt = (
            "You are Dwarpal, an Indian telephony fraud analyst. "
            "Read the full multi-turn phone call transcript (Gujarati/Hindi/English) and decide if the caller "
            "is attempting a scam (Digital Arrest, FIR/Police/CBI/Customs threats, Bank KYC/OTP fraud, "
            "UPI collect traps, AnyDesk) or having a normal harmless conversation (friends, college, family, work). "
            'Respond ONLY with a single JSON object: {"is_scam": true or false, "confidence": 0.0 to 1.0, '
            '"warning": "One clear sentence explaining why this call is a scam or safe."}'
        )
        user_prompt = (
            f"Behavioral signals flagged so far: {active_dims}\n"
            f"Full conversation (last {len(past_10_turns)} turns):\n{formatted_turns}\n\n"
            "JSON:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + '{"is_scam":'

        inputs = self.llm_tokenizer(prompt_text, return_tensors="pt").to(self.llm_device)

        with self._llm_lock, torch.no_grad():
            output_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=85,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )

        generated = '{"is_scam":' + self.llm_tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        verdict = {
            "is_scam": True,
            "confidence": 0.78,
            "warning": "High-pressure social engineering detected across conversation turns.",
        }
        try:
            json_match = re.search(r"\{.*?\}", generated, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                is_scam = bool(parsed.get("is_scam", False))
                warning = str(parsed.get("warning", verdict["warning"]))
                if not is_scam and any(w in warning.lower() for w in ["scammer", "fraud", "deceive", "threaten", "phishing"]):
                    is_scam = True
                verdict = {
                    "is_scam": is_scam,
                    "confidence": float(parsed.get("confidence", 0.80)),
                    "warning": warning,
                }
        except Exception:
            pass

        state = self.call_states.get(call_id)
        if not state:
            return None

        state.llm_in_flight = False
        state.last_llm_verdict = verdict

        should_alert = False
        if verdict["is_scam"]:
            state.confirmed_scam = True
            final_score = round(max(state.peak_score, verdict["confidence"], 0.88), 3)
            state.peak_score = final_score
            if not state.alert_triggered:
                state.alert_triggered = True
                should_alert = True
            level = "CRITICAL_SCAM"
        else:
            state.confirmed_scam = False
            state.cooldown_turns_remaining = 2
            final_score = 0.25
            level = "SAFE"

        return {
            "risk_score": final_score,
            "level": level,
            "is_scam": verdict["is_scam"],
            "confidence": verdict["confidence"],
            "warning": verdict["warning"],
            "alert": should_alert,
        }
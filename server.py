import asyncio
import base64
import json
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn
from scam_detector import HybridScamDetector
from stt_engine import CallSession, SharedSTTEngine

app = FastAPI(title="Dwarpal Two-Way Call Bridge & Scam Detection Server")

# ==============================================================================
# CALL ROUTING CONFIGURATION
# ==============================================================================
TWILIO_CALLER_ID = ""       # Your Twilio number (valid E.164 callerId for <Dial>)
VICTIM_PHONE_NUMBER = ""    # Your phone (Victim B) receiving the bridged call
MOTHER_TEST_NUMBER = ""     # Caller A test number

# Set to True later when routing directly to your Android Twilio Voice SDK client
USE_VOIP_CLIENT = False
ANDROID_CLIENT_IDENTITY = "dwarpal_android_user"
# ==============================================================================

stt_engine = SharedSTTEngine(model_id="bodhan-ai/indic-transcribe-flex", device="cuda")

scam_detector = HybridScamDetector(
    nli_model_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    nli_device="cpu",
    llm_device="cuda",
    shared_gpu_lock=stt_engine._gpu_lock,
)

# Maps both StreamSid (MZ...) and CallSid (CA...) to WebSocket subscribers & latest verdicts
alert_subscribers: dict[str, set[WebSocket]] = {}
latest_call_verdicts: dict[str, dict] = {}
stream_to_call_sid: dict[str, str] = {}
call_to_stream_sid: dict[str, str] = {}


def _resolve_keys(identifier: str) -> list[str]:
    """Returns all linked IDs (StreamSid and CallSid) for a given call identifier."""
    keys = {identifier}
    if identifier in stream_to_call_sid:
        keys.add(stream_to_call_sid[identifier])
    if identifier in call_to_stream_sid:
        keys.add(call_to_stream_sid[identifier])
    return list(keys)


async def broadcast_alert(stream_sid: str, payload: dict):
    """Pushes live transcript & risk verdicts to subscribers listening on StreamSid OR CallSid."""
    keys = _resolve_keys(stream_sid)
    for k in keys:
        latest_call_verdicts[k] = payload

    message_text = json.dumps(payload, ensure_ascii=False)
    for k in keys:
        subscribers = alert_subscribers.get(k)
        if not subscribers:
            continue
        dead_sockets = []
        for ws in list(subscribers):
            try:
                await ws.send_text(message_text)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            subscribers.discard(ws)


@app.post("/")
@app.post("/incoming-call")
async def handle_incoming_call(
    request: Request,
    CallSid: str = Form(default="unknown"),
    From: str = Form(default="unknown"),
):
    """
    Twilio Webhook for incoming calls:
      1. Forks ONLY Caller A's voice (inbound_track) to /media-stream via <Start><Stream>
      2. Bridges the live two-way call to Victim B's phone or Android VoIP app via <Dial>
    """
    host = request.headers.get("host")
    caller_tag = "MOTHER_TEST_PHONE (Caller A)" if From.endswith("") else From
    print(f"\n[📞 INCOMING CALL] From: {caller_tag} | CallSid: {CallSid}")

    if USE_VOIP_CLIENT:
        dial_target = f"<Client>{ANDROID_CLIENT_IDENTITY}</Client>"
    else:
        dial_target = f"<Number>{VICTIM_PHONE_NUMBER}</Number>"

    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="wss://{host}/media-stream" track="inbound_track" />
    </Start>
    <Dial callerId="{TWILIO_CALLER_ID}">
        {dial_target}
    </Dial>
</Response>"""
    return Response(content=twiml_response, media_type="text/xml")


@app.get("/api/status/{call_or_stream_sid}")
async def get_call_status(call_or_stream_sid: str):
    """REST polling endpoint accepting either Twilio CallSid (CA...) or StreamSid (MZ...)."""
    for k in _resolve_keys(call_or_stream_sid):
        if k in latest_call_verdicts:
            return JSONResponse(content=latest_call_verdicts[k])

    return JSONResponse(
        content={
            "id": call_or_stream_sid,
            "status": "NO_DATA",
            "risk_score": 0.0,
            "level": "SAFE",
            "alert": False,
        }
    )


@app.websocket("/ws/alerts/{call_or_stream_sid}")
async def handle_client_alerts(websocket: WebSocket, call_or_stream_sid: str):
    """
    Real-time WebSocket feed for the Android VoIP app.
    Android connects using Twilio's CallSid (CA...) as soon as the phone rings.
    """
    await websocket.accept()
    alert_subscribers.setdefault(call_or_stream_sid, set()).add(websocket)
    print(f"[+] Alert UI Client Subscribed | ID: {call_or_stream_sid}")

    try:
        for k in _resolve_keys(call_or_stream_sid):
            if k in latest_call_verdicts:
                await websocket.send_text(json.dumps(latest_call_verdicts[k], ensure_ascii=False))
                break

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if call_or_stream_sid in alert_subscribers:
            alert_subscribers[call_or_stream_sid].discard(websocket)
            if not alert_subscribers[call_or_stream_sid]:
                alert_subscribers.pop(call_or_stream_sid, None)
        print(f"[-] Alert UI Client Disconnected | ID: {call_or_stream_sid}")


async def tier3_background_task(
    stream_sid: str,
    call_sid: str,
    past_10_turns: list[dict],
    active_tactics: list[str],
):
    """Runs Tier 3 (Qwen2.5-1.5B) asynchronously on the full past 10 turns."""
    loop = asyncio.get_running_loop()
    print(f"[TIER 3 ESCALATION | {call_sid}] Analyzing past {len(past_10_turns)} turns in background...")

    llm_result = await loop.run_in_executor(
        None,
        scam_detector.run_tier3_deliberation,
        stream_sid,
        past_10_turns,
        active_tactics,
    )
    if not llm_result:
        return

    print(
        f"[TIER 3 VERDICT | {call_sid}] Scam: {llm_result['is_scam']} | "
        f"Level: {llm_result['level']} | Score: {llm_result['risk_score']} | "
        f"Reason: {llm_result['warning']}"
    )

    if llm_result["alert"]:
        print(f"\n🚨 [SCAM ALERT TRIGGERED BY TIER 3 | {call_sid}] {llm_result['warning']}\n")
    elif not llm_result["is_scam"]:
        print(f"[TIER 3 LOOP-BACK | {call_sid}] Cleared as benign — returning control to Tier 1+2.\n")

    await broadcast_alert(
        stream_sid,
        {
            "stream_sid": stream_sid,
            "call_sid": call_sid,
            "source": "TIER3_LLM",
            "risk_score": llm_result["risk_score"],
            "level": llm_result["level"],
            "active_tactics": active_tactics,
            "alert": llm_result["alert"],
            "warning": llm_result["warning"],
        },
    )


async def inference_worker(
    stream_sid: str,
    call_sid: str,
    sessions: dict[str, CallSession],
    segment_queue: asyncio.Queue,
):
    loop = asyncio.get_running_loop()

    while True:
        item = await segment_queue.get()
        if item is None:
            segment_queue.task_done()
            break

        track, audio_pcm = item
        try:
            result = await loop.run_in_executor(
                None, stt_engine.transcribe_audio, audio_pcm, "mixed"
            )
            if result:
                session = sessions[track]
                session.record_turn(result)
                print(f"\n[CALLER A TRANSCRIPT | {call_sid}] {result['formatted']}")

                fast_verdict = await loop.run_in_executor(
                    None,
                    scam_detector.analyze_fast_tiers,
                    stream_sid,
                    session.transcript_history,
                )
                print(
                    f"[TIER 1+2 FAST] Level: {fast_verdict['level']} | "
                    f"Score: {fast_verdict['risk_score']} "
                    f"(Behavior: {fast_verdict['behavioral_score']}, Semantic: {fast_verdict['semantic_risk']}) | "
                    f"Tactics: {fast_verdict['active_tactics']}"
                )

                if fast_verdict["alert"]:
                    banner = fast_verdict["warning"] or "High-confidence social engineering detected!"
                    print(f"\n🚨 [FAST ALERT TRIGGERED | {call_sid}] {banner}\n")

                if fast_verdict["trigger_tier3"]:
                    state = scam_detector.get_or_create_state(stream_sid)
                    state.llm_in_flight = True
                    past_10_snapshot = list(session.transcript_history[-10:])
                    asyncio.create_task(
                        tier3_background_task(
                            stream_sid,
                            call_sid,
                            past_10_snapshot,
                            list(fast_verdict["active_tactics"]),
                        )
                    )

                await broadcast_alert(
                    stream_sid,
                    {
                        "stream_sid": stream_sid,
                        "call_sid": call_sid,
                        "source": "TIER1_2_FAST",
                        "track": track,
                        "lang": result["lang"],
                        "transcript": result["text"],
                        "risk_score": fast_verdict["risk_score"],
                        "level": fast_verdict["level"],
                        "active_tactics": fast_verdict["active_tactics"],
                        "alert": fast_verdict["alert"],
                        "warning": fast_verdict["warning"],
                    },
                )
        except Exception as e:
            print(f"\n[!] Inference Error ({call_sid} / {track}): {e}")
        finally:
            segment_queue.task_done()


def _cleanup_call_state(stream_sid: str, call_sid: str):
    scam_detector.clear_call(stream_sid)
    for k in _resolve_keys(stream_sid):
        latest_call_verdicts.pop(k, None)
    stream_to_call_sid.pop(stream_sid, None)
    call_to_stream_sid.pop(call_sid, None)


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] Twilio Media Stream WebSocket Connected!")

    stream_sid = "unknown"
    call_sid = "unknown"
    sessions: dict[str, CallSession] = {}
    segment_queue: asyncio.Queue = asyncio.Queue()
    worker_task: asyncio.Task | None = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event_type = data.get("event")

            if event_type == "start":
                start_meta = data.get("start", {})
                stream_sid = data.get("streamSid", start_meta.get("streamSid", "unknown"))
                call_sid = start_meta.get("callSid", "unknown")

                stream_to_call_sid[stream_sid] = call_sid
                call_to_stream_sid[call_sid] = stream_sid

                sessions.clear()
                worker_task = asyncio.create_task(
                    inference_worker(stream_sid, call_sid, sessions, segment_queue)
                )
                print(f"[+] Two-Way Bridge Started | CallSid: {call_sid} | StreamSid: {stream_sid}")

            elif event_type == "media":
                track = data["media"].get("track", "inbound")
                if track not in sessions:
                    sessions[track] = stt_engine.create_session(
                        call_id=stream_sid, track=track
                    )

                raw_audio_bytes = base64.b64decode(data["media"]["payload"])
                ready_segments = sessions[track].ingest_chunk(raw_audio_bytes)
                for seg in ready_segments:
                    await segment_queue.put((track, seg))

            elif event_type == "stop":
                for track, session in sessions.items():
                    final_seg = session.flush_buffer()
                    if final_seg is not None:
                        await segment_queue.put((track, final_seg))

                await segment_queue.put(None)
                if worker_task:
                    await worker_task
                _cleanup_call_state(stream_sid, call_sid)
                print(f"\n[-] Call Stream Ended | CallSid: {call_sid}")
                break

    except WebSocketDisconnect:
        print(f"\n[-] Client Disconnected | CallSid: {call_sid}")
    except Exception as e:
        print(f"\n[!] Stream Error ({call_sid}): {e}")
    finally:
        _cleanup_call_state(stream_sid, call_sid)
        if worker_task and not worker_task.done():
            await segment_queue.put(None)
            await worker_task


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
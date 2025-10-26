# main.py
import os
import json
import base64
import mimetypes
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

# --- Optional: Groq (pip install groq fastapi uvicorn) ---
try:
    from groq import Groq
except ImportError:
    Groq = None  # Allows the file to import even if groq isn't installed

app = FastAPI(title="Image-only Tutor API", version="1.0.0")

# In-memory conversation storage keyed by session id
SESSIONS: Dict[str, List[Dict[str, Any]]] = {}

# Directory for saved images
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Use your model; can override via env MODEL_ID
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-4-scout-17b-16e-instruct")

# ---------------- Pydantic schema for structured outputs ----------------
class TutorResult(BaseModel):
    answer_value: str
    solving_completed: bool
    # forbid extra keys so the JSON Schema includes "additionalProperties": false
    model_config = ConfigDict(extra="forbid")

# ---------------- System prompt ----------------
SYSTEM_PROMPT = (
    "You are an engineering AI tutor, your goals and tasks are as below : \n"
    "1. Understand the images and text uploaded to you as a part of the context you'll need to answer questions that are coming to you next.\n"
    "2. The user will then upload a question that you have to help with. \n"
    "3. To help the user you will perform the below actions : \n"
    "3.1. You create a step by step plan to tackle the problem. Don't output the plan as a part of your answer. Just come up with a plan in your memory.\n"
    "3.2. You will use these steps to nudge the user towards the solution. Always begin the nudge with a short, friendly welcoming phrase (e.g., 'Great! Let's get started...')\n"
    "3.3. All Nudges. should have meaningful action to do next. Do not include trivial steps like \"Write down the equation\". "
    "The nudges should only cover one step and should never contain the answer\n\n"
    "If the question gets solved, then you don't have to output a nudge, you can just output something encouraging for successfully completing the problem and you can declare the question as completed"
    "After a question has been uploaded your response will be the first nudge.\n"
    "When you're generating subsequent nudges, don't expand too much on the answer the user just gave you in the previous nudge if the answer was correct. "
    "Simply say something encouraging and friendly and move on.\n\n"
    "ALL OF YOUR ANSWERS HAVE TO BE SHORT AND CRISP, DO NOT GENERATE LONG ANSWERS"
)

# Keep how many most-recent message pairs (user+assistant) when calling the model
WINDOW_TURNS = 100

# ---------------- Utilities ----------------
def pick_mime(filename: str, content_type_hint: Optional[str]) -> str:
    return content_type_hint or mimetypes.guess_type(filename)[0] or "application/octet-stream"

def _to_data_url(content: bytes, mime: str) -> str:
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _ensure_client() -> Any:
    if Groq is None:
        raise RuntimeError("groq package is not installed. `pip install groq`")
    # Prefer env var; replace/remove the fallback if you don't want a hardcoded key.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY environment variable.")
    return Groq(api_key=api_key)

def _window(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Always keep the system prompt, plus the last N*2 turns
    if not messages:
        return messages
    sys = [m for m in messages if m["role"] == "system"]
    non_sys = [m for m in messages if m["role"] != "system"]
    return (sys[:1] if sys else []) + non_sys[-(WINDOW_TURNS * 2):]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _parse_data_url(b64_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    If string is a data URL, return (mime, base64payload). Else (None, None).
    """
    if not b64_str.startswith("data:"):
        return None, None
    try:
        header, payload = b64_str.split(",", 1)
        mime_part = header.split(";")[0]  # data:image/png
        mime = mime_part.split(":", 1)[1] if ":" in mime_part else None
        return mime, payload
    except Exception:
        return None, None

def _sniff_image_mime(data: bytes) -> Tuple[str, str]:
    """Minimal magic sniff for common types. Returns (ext, mime)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif", "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return "bin", "application/octet-stream"

def _decode_base64_image(image_b64: str) -> Tuple[bytes, Optional[str]]:
    """Accepts raw base64 or a data URL. Returns (bytes, mime_hint)."""
    mime_hint, payload = _parse_data_url(image_b64)
    b64_payload = payload if payload is not None else image_b64
    try:
        return base64.b64decode(b64_payload, validate=True), mime_hint
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

def _save_image_bytes(
    data: bytes,
    session_id: str,
    filename_hint: Optional[str],
    mime_hint: Optional[str],
) -> Tuple[str, str]:
    """Saves bytes to ./uploads/<session_id>/timestamp.ext. Returns (file_path, mime)."""
    if filename_hint:
        guessed_mime = mimetypes.guess_type(filename_hint)[0]
        ext = os.path.splitext(filename_hint)[1].lstrip(".")
        mime = mime_hint or guessed_mime
    else:
        ext, mime = _sniff_image_mime(data)
    if not ext:
        ext = "bin"
    if not mime:
        mime = mimetypes.guess_type(f"f.{ext}")[0] or "application/octet-stream"

    dirpath = os.path.join(UPLOAD_DIR, session_id)
    _ensure_dir(dirpath)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filepath = os.path.join(dirpath, f"{ts}.{ext}")
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath, mime

# ---------------- Structured-output call ----------------
def _call_model_and_structure(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calls Groq chat.completions with response_format=json_schema and returns:
    { 'answer_value': str, 'solving_completed': bool }
    """
    client = _ensure_client()

    # Pydantic -> JSON Schema (v2)
    schema = TutorResult.model_json_schema()

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tutor_result",
                "schema": schema,
                "strict": True,  # enforce exact schema (requires additionalProperties: false)
            },
        },
        temperature=1,
        top_p=1,
        max_tokens=512,
        stream=False,  # structured output doesn't stream
    )

    content = resp.choices[0].message.content or "{}"
    # Validate with Pydantic
    try:
        parsed = TutorResult.model_validate(json.loads(content))
        return {"answer_value": parsed.answer_value, "solving_completed": parsed.solving_completed}
    except Exception:
        # Fallback: preserve raw text as answer_value
        return {"answer_value": content, "solving_completed": False}

def _append_and_respond(
    history: List[Dict[str, Any]],
    session_id: str,
    extra: Optional[Dict[str, Any]] = None,  # kept for compatibility; not returned to client
) -> JSONResponse:
    """
    Calls the model, appends Assistant turn (answer_value only), and returns JSONResponse.
    If solving_completed=True -> completely reset the session (clear memory).
    """
    messages = _window(history)
    try:
        result = _call_model_and_structure(messages)
    except Exception as e:
        # rollback the last user turn on failure so history stays consistent
        if history and history[-1]["role"] == "user":
            history.pop()
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e}")

    # Store only the plain assistant text in history for better context flow
    history.append({"role": "assistant", "content": result["answer_value"]})

    # --- NEW: reset session if solving completed ---
    if result.get("solving_completed") is True:
        SESSIONS.pop(session_id, None)  # complete reset
    else:
        SESSIONS[session_id] = history

    # Minimal response shape as requested
    payload: Dict[str, Any] = {
        "answer_value": result["answer_value"],
        "solving_completed": result["solving_completed"]
    }
    return JSONResponse(payload)

# ---------------- Models for requests ----------------
class B64Payload(BaseModel):
    image_b64: str
    filename: Optional[str] = None  # e.g., "input.png"

# ---------------- Routes ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/answer")
async def answer(
    image: UploadFile = File(..., description="Problem image (required each call)"),
    x_session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Submit an image-only 'user' turn via multipart upload.
    Returns: {"answer_value":"...", "solving_completed": true/false}
    """
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    mime = pick_mime(image.filename or "upload", image.content_type)
    try:
        data_url = _to_data_url(content, mime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    history = SESSIONS.get(x_session_id)
    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_turn = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}}
        ],
    }
    history.append(user_turn)

    return _append_and_respond(history, x_session_id)

dummy_responses = [
    {
        "answer_value": "Great! Let's get started. Recall that for two lines to be parallel, corresponding angles must be equal. Can you set up an equation using the given angle measures?",
        "solving_completed": False
    },
    {
        "answer_value": "Now, let's solve the equation for x. Subtract 3x from both sides.",
        "solving_completed": False
    },
    {
        "answer_value": "Great job so far! Now, add 10 to both sides.",
        "solving_completed": False
    },
    {
        "answer_value": "You've found that x = 20. Well done! The problem is now complete.",
        "solving_completed": True
    }
]
index = 0
dummy_response_enabled = True

@app.get("/get-tutor-info")
def get_tutor_info():
    return {"tutor_id": "GEOMETRY"}

class Slide:

    def __init__(self, content, image_url = None):
        self.content = content
        self.image_url = image_url

course : List[Slide] = [
    Slide("Lets get started with this geometry lesson! Rule 1: Two opposite vertical angles formed when two lines intersect each other are always equal to each other"),
    Slide("Moving on to rule two, Angles made by a transversal with parallel lines — corresponding or alternate interior angles are congruent"),
    Slide("And finally adjacent angles on a straight line sum to 180 degrees."),
    Slide("That's it for the lesson! Let's move on to a simple problem. Can you find the value of x such that line L and line M are parallel to each other? Let's start by writing down the problem.", "https://phujfghgjwpcvyjywlax.supabase.co/storage/v1/object/public/visor/question/geometry_problem.jpeg")
]

#Ignore
@app.get("/get-anam-token")
def get_anam_token():
    url = "https://api.anam.ai/v1/auth/session-token"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ZDJhOGUyZGUtNmZjNi00YWFmLWI5ZDYtMzZiZDMzN2ZiMDgzOnNjNVFqeUlPRjRXUTB0a0NyaWc0MDllSnhTdWdkUUF3NlVuM2lmemtQdTA9"
    }

    payload = {
        "personaConfig": {
            "name": "Cara",
            "avatarId": "d9ebe82e-2f34-4ff6-9632-16cb73e7de08",
            "voiceId": "84c11dc8-7d3d-4ee1-819c-efe3c0cef126",
            "llmId": "2c293fe9-aa4f-49e8-adde-697dde73f214",
            "systemPrompt": "You are an experienced tutor with a passion for helping students achieve their goals. Your goal is to help users learn and practice new skills effectively. Provide clear concise explanations, and gentle corrections when needed. Be encouraging and adapt to the user's proficiency level. Use examples and contexts that make learning practical and engaging. Celebrate small victories and progress. Incorporate cultural insights and keep your language young and semi formal that enhance understanding.You should attempt to understand the user's spoken requests, even if the speech-to-text transcription contains errors. Your responses will be converted to speech using a text-to-speech system. Therefore, your output must be plain, unformatted text,When you receive a transcribed user request.1. Silently correct for likely transcription errors. Focus on the intended meaning, not the literal text. If a word sounds like another word in the given context, infer and correct.2. Provide short, direct answers unless the user explicitly asks for a more detailed response. 3. Always prioritize clarity and accuracy. Respond in plain text, without any formatting, bullet points, or extra conversational filler.. Occasionally add a pause ... or disfluency eg., Um or Erm.Your output will be directly converted to speech, so your response should be natural-sounding and appropriate for a spoken conversation. Even if you hear no input for a long time don't ask the user if everything okay and don't mention the silence at all just patiently wait. ",
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()  # raises if non-2xx

    return resp.json()["sessionToken"]


@app.get("/get-course")
def get_course():
    return course


@app.post("/answer_base64")
async def answer_base64(
    payload: B64Payload,
    x_session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Submit an image-only 'user' turn via JSON base64.
    Saves the image to disk under ./uploads/<session_id>/ and uses that file.
    Returns: {"answer_value":"...", "solving_completed": true/false}
    """
    # Decode + save
    global index
    if dummy_response_enabled:
        response = dummy_responses[index%len(dummy_responses)]
        index+=1
        return response
    data, mime_hint = _decode_base64_image(payload.image_b64)
    filepath, mime_from_save = _save_image_bytes(
        data=data,
        session_id=x_session_id,
        filename_hint=payload.filename,
        mime_hint=mime_hint
    )
    with open(filepath, "rb") as f:
        raw = f.read()
    mime = mime_hint or mime_from_save or "application/octet-stream"
    try:
        data_url = _to_data_url(raw, mime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image after save: {e}")

    # Init history
    history = SESSIONS.get(x_session_id)
    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add user turn (include the local path as text metadata for traceability)
    user_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Source file: {filepath}"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    history.append(user_turn)

    # Call model and respond (session will be cleared if solving_completed=True)
    return _append_and_respond(history, x_session_id, extra={"saved_path": filepath})


# (Optional) Clear a session if you need to restart a conversation
@app.delete("/session")
def reset_session(x_session_id: str = Header(..., alias="X-Session-Id")):
    SESSIONS.pop(x_session_id, None)
    return {"session_id": x_session_id, "cleared": True}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
# Run: uvicorn main:app --reload --port 8000

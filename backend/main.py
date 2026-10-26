import ast
import asyncio
import base64
import io
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq, RateLimitError
from PIL import Image, ImageOps

load_dotenv()

MODEL_NAME = "qwen/qwen3.8-27b"
MAX_IMAGE_SIDE = 2048
MIN_IMAGE_SIDE = 32
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

FULL_PAGE_PROMPT = (
    "Please output only the text content from the image without any additional "
    "descriptions or formatting."
)
SPOTTING_PROMPT = (
    "Spotting all the text in the image with line-level, and output in JSON format "
    "with keys 'text_content' and 'bbox_2d'."
)

app = FastAPI(title="Qwen 3.8 Vision OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client() -> AsyncGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in .env")
    return AsyncGroq(api_key=api_key)


async def prepare_image(file: UploadFile) -> tuple[str, int, int]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, or WEBP images are supported")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large (max 15 MB)")

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    if image.mode != "RGB":
        image = image.convert("RGB")

    if min(image.size) < MIN_IMAGE_SIDE:
        raise HTTPException(status_code=400, detail=f"Image must be at least {MIN_IMAGE_SIDE}px on each side")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, image.width, image.height


def build_messages(base64_image: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        },
    ]


async def create_raw_completion(client: AsyncGroq, messages: list[dict], stream: bool):
    max_tokens = 900
    for attempt in range(3):
        try:
            return await client.chat.completions.with_raw_response.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=max_tokens,
                top_p=0.95,
                reasoning_effort="default",
                stream=stream,
                stop=None,
            )
        except RateLimitError as err:
            if attempt == 2:
                raise
            msg = str(err)
            match = re.search(r"Limit\s+(\d+),\s+Requested\s+(\d+)", msg)
            if match:
                limit_val = int(match.group(1))
                max_tokens = max(256, min(max_tokens, limit_val - 50))
            await asyncio.sleep(2.5 * (attempt + 1))


def extract_rate_limits(headers) -> dict:
    def to_int(key: str):
        val = headers.get(key)
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "limit_tokens": to_int("x-ratelimit-limit-tokens"),
        "remaining_tokens": to_int("x-ratelimit-remaining-tokens"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
        "limit_requests": to_int("x-ratelimit-limit-requests"),
        "remaining_requests": to_int("x-ratelimit-remaining-requests"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
    }


def extract_usage(usage_obj) -> dict | None:
    if not usage_obj:
        return None
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
        "total_tokens": getattr(usage_obj, "total_tokens", 0),
        "total_time": round(float(getattr(usage_obj, "total_time", 0.0) or 0.0), 3),
    }


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_json(raw: str):
    text = strip_think(raw)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    brackets = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if brackets:
        text = brackets.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            objects = []
            for match in re.finditer(r"\{[^{}]*\}", text):
                try:
                    obj = json.loads(match.group(0))
                    if isinstance(obj, dict) and ("bbox_2d" in obj or "bbox" in obj):
                        objects.append(obj)
                except Exception:
                    continue
            return objects if objects else None


def normalize_boxes(parsed, width: int, height: int) -> list[dict]:
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    if not isinstance(parsed, list):
        return []

    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = item.get("bbox_2d") or item.get("bbox")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        try:
            coords = [float(c) for c in coords]
        except (TypeError, ValueError):
            continue
        items.append((str(item.get("text_content", item.get("text", ""))), coords))

    if not items:
        return []

    half = len(items) // 2
    if len(items) % 2 == 0 and half > 0 and [t for t, _ in items[:half]] == [t for t, _ in items[half:]]:
        items = items[:half]

    longest = float(max(width, height))
    scale_x = 1000.0 * width / longest
    scale_y = 1000.0 * height / longest

    filtered = []
    for idx, (text, coords) in enumerate(items):
        x1, y1, x2, y2 = coords
        if idx > 0 and (max(x1, x2) > scale_x * 1.08 or max(y1, y2) > scale_y * 1.08):
            continue
        filtered.append((text, coords))

    if filtered:
        items = filtered

    max_value = max(max(coords) for _, coords in items)
    if max_value <= 1.0:
        sx, sy = 1.0, 1.0
    elif max_value <= 1000.0:
        sx, sy = scale_x, scale_y
    else:
        sx, sy = float(width), float(height)

    boxes = []
    for text, (x1, y1, x2, y2) in items:
        nx1, nx2 = sorted((x1 / sx, x2 / sx))
        ny1, ny2 = sorted((y1 / sy, y2 / sy))
        boxes.append({
            "text": text,
            "box": [
                round(min(max(nx1, 0.0), 1.0), 5),
                round(min(max(ny1, 0.0), 1.0), 5),
                round(min(max(nx2, 0.0), 1.0), 5),
                round(min(max(ny2, 0.0), 1.0), 5),
            ],
        })
    return boxes


async def stream_events(completion, rate_limit: dict):
    yield json.dumps({"type": "rate_limit", "data": rate_limit}) + "\n"

    buffer = ""
    in_think = False
    open_tag, close_tag = "<think>", "</think>"
    usage_data = None

    async for chunk in completion:
        x_groq = getattr(chunk, "x_groq", None)
        if x_groq and getattr(x_groq, "usage", None):
            usage_data = extract_usage(x_groq.usage)

        if not chunk.choices:
            continue
        buffer += chunk.choices[0].delta.content or ""

        while buffer:
            if in_think:
                end = buffer.find(close_tag)
                if end == -1:
                    buffer = buffer[-(len(close_tag) - 1):]
                    break
                buffer = buffer[end + len(close_tag):].lstrip()
                in_think = False
            else:
                start = buffer.find(open_tag)
                if start != -1:
                    if start > 0:
                        yield json.dumps({"type": "token", "text": buffer[:start]}) + "\n"
                    buffer = buffer[start + len(open_tag):]
                    in_think = True
                    continue
                safe = len(buffer)
                for i in range(1, len(open_tag)):
                    if buffer.endswith(open_tag[:i]):
                        safe = len(buffer) - i
                        break
                if safe > 0:
                    yield json.dumps({"type": "token", "text": buffer[:safe]}) + "\n"
                buffer = buffer[safe:]
                break

    if buffer and not in_think:
        yield json.dumps({"type": "token", "text": buffer}) + "\n"

    if usage_data:
        yield json.dumps({"type": "usage", "data": usage_data}) + "\n"


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
    }


@app.post("/api/ocr/full")
async def ocr_full(file: UploadFile = File(...)):
    client = get_client()
    base64_image, _, _ = await prepare_image(file)

    try:
        raw_response = await create_raw_completion(
            client, build_messages(base64_image, FULL_PAGE_PROMPT), stream=True
        )
        rate_limit = extract_rate_limits(raw_response.headers)
        completion = await raw_response.parse()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    async def generator():
        try:
            async for event in stream_events(completion, rate_limit):
                yield event
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")


@app.post("/api/ocr/spotting")
async def ocr_spotting(file: UploadFile = File(...)):
    client = get_client()
    base64_image, width, height = await prepare_image(file)

    try:
        raw_response = await create_raw_completion(
            client, build_messages(base64_image, SPOTTING_PROMPT), stream=False
        )
        rate_limit = extract_rate_limits(raw_response.headers)
        completion = await raw_response.parse()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    raw = completion.choices[0].message.content or ""
    boxes = normalize_boxes(parse_json(raw), width, height)
    usage = extract_usage(getattr(completion, "usage", None))

    return {
        "boxes": boxes,
        "raw": strip_think(raw),
        "width": width,
        "height": height,
        "usage": usage,
        "rate_limit": rate_limit,
    }

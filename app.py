import streamlit as st
import json
import re
import ast
from PIL import Image, ImageDraw, ImageFont
import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Qwen 3.8 OCR (Groq)", layout="wide")

st.sidebar.title("Qwen 3.8 OCR")
st.sidebar.markdown("Powered by **Groq** (`qwen/qwen3.8-27b`)")

env_api_key = os.getenv("GROQ_API_KEY", "").strip()
if env_api_key:
    api_key = env_api_key
    st.sidebar.success("GROQ_API_KEY terdeteksi dari `.env`")
else:
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    ).strip()
    if not api_key:
        st.sidebar.warning("Masukkan GROQ_API_KEY di file `.env` atau kolom di atas.")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

temp_image_path = "temp_uploaded_image.jpg"
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
    
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    image.save(temp_image_path)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_json(json_output):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_output)
    if match:
        json_output = match.group(1).strip()

    match_brackets = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", json_output)
    if match_brackets:
        json_output = match_brackets.group(1).strip()

    try:
        return json.loads(json_output)
    except Exception:
        try:
            return ast.literal_eval(json_output)
        except Exception:
            return None


def inference_with_groq_stream(api_token, image_path, prompt, sys_prompt="You are a helpful assistant."):
    base64_image = encode_image(image_path)
    client = Groq(api_key=api_token)

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    })

    completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="default",
        stream=True,
        stop=None
    )

    return completion


def plot_text_bounding_boxes(image_path, raw_boxes):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    boxes = parse_json(raw_boxes)
    if not boxes or not isinstance(boxes, list):
        return img

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    def scale_coord(val, max_size):
        if val <= 1.0:
            return int(val * max_size)
        elif val <= 1000.0:
            return int((val / 1000.0) * max_size)
        return int(val)

    color = '#00FF00'

    for bounding_box in boxes:
        if not isinstance(bounding_box, dict) or "bbox_2d" not in bounding_box:
            continue

        coords = bounding_box["bbox_2d"]
        if len(coords) != 4:
            continue

        abs_x1 = scale_coord(coords[0], width)
        abs_y1 = scale_coord(coords[1], height)
        abs_x2 = scale_coord(coords[2], width)
        abs_y2 = scale_coord(coords[3], height)

        abs_x1, abs_x2 = min(abs_x1, abs_x2), max(abs_x1, abs_x2)
        abs_y1, abs_y2 = min(abs_y1, abs_y2), max(abs_y1, abs_y2)

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=2)

        text_content = bounding_box.get("text_content", "")
        if text_content:
            draw.text((abs_x1, max(0, abs_y1 - 15)), text_content[:30], fill=color, font=font)

    return img


st.title("Qwen 3.8 Vision OCR")
st.markdown("Aplikasi OCR berbasis AI menggunakan model **Qwen 3.8 27B** via **Groq**.")

mode = st.radio("Select Mode", ["Full Page OCR", "Text Spotting"], horizontal=True)

if uploaded_file is not None:
    if mode == "Full Page OCR":
        st.header("Full Page OCR")
        st.caption("Mengekstrak seluruh teks yang ada pada gambar dokumen.")

        if st.button("Extract Text", type="primary"):
            if not api_key:
                st.error("Silakan masukkan GROQ_API_KEY terlebih dahulu di file `.env` atau sidebar.")
            else:
                prompt = "Please output only the text content from the image without any additional descriptions or formatting."
                try:
                    stream_obj = inference_with_groq_stream(api_key, temp_image_path, prompt)
                    st.markdown("### Extracted Text:")
                    
                    def stream_generator():
                        for chunk in stream_obj:
                            delta = chunk.choices[0].delta.content or ""
                            yield delta

                    st.write_stream(stream_generator)
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi API: {str(e)}")

    elif mode == "Text Spotting":
        st.header("Text Spotting")
        st.caption("Mendeteksi posisi teks baris per baris beserta koordinat bounding box.")

        if st.button("Spot Text", type="primary"):
            if not api_key:
                st.error("Silakan masukkan GROQ_API_KEY terlebih dahulu di file `.env` atau sidebar.")
            else:
                prompt = "Spotting all the text in the image with line-level, and output in JSON format with keys 'text_content' and 'bbox_2d'."
                try:
                    with st.spinner("Sedang memproses deteksi teks..."):
                        stream_obj = inference_with_groq_stream(api_key, temp_image_path, prompt)
                        full_response = ""
                        for chunk in stream_obj:
                            full_response += chunk.choices[0].delta.content or ""

                        result_image = plot_text_bounding_boxes(temp_image_path, full_response)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(result_image, caption="Text Spotting Result", use_container_width=True)
                    with col2:
                        st.markdown("### Detected Text & JSON:")
                        st.code(full_response, language="json")
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi API: {str(e)}")
else:
    st.info("Silakan unggah gambar di sidebar sebelah kiri untuk mulai.")

if os.path.exists(temp_image_path) and uploaded_file is None:
    try:
        os.remove(temp_image_path)
    except Exception:
        pass
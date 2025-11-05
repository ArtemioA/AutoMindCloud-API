import os, base64, binascii
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from openai import OpenAI

# Lee la API key desde el entorno (Secret Manager en Cloud Run)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")   # modelo con visión
MAX_REQ_BYTES = 32 * 1024 * 1024             # ~32 MiB

client = OpenAI()
app = FastAPI(title="GPT Proxy", version="2.0-vision")

class InferenceIn(BaseModel):
    text: str = Field(..., description="Prompt para el modelo")
    image_b64: Optional[str] = Field(None, description="Imagen en base64 (SIN prefijo data:)")
    mime: Optional[str] = Field(None, description="image/jpeg | image/png | image/webp, etc.")

class InferenceOut(BaseModel):
    model: str
    output: str
    debug: Optional[Dict[str, Any]] = None  # útil para ver tamaños, mime, etc.

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        has_img = bool(payload.image_b64 and payload.image_b64.strip())
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": payload.text}]

        debug: Dict[str, Any] = {
            "has_image_b64": has_img,
            "mime": payload.mime,
            "approx_bytes": None,
            "decoded_len": None,
        }

        if has_img:
            # tamaño aprox (base64 es ~4/3 del binario)
            approx_bytes = int(len(payload.image_b64) * 0.75)
            debug["approx_bytes"] = approx_bytes
            if approx_bytes > MAX_REQ_BYTES:
                raise HTTPException(status_code=413, detail="Imagen demasiado grande (~>32 MiB).")

            # validar base64
            try:
                img_bytes = base64.b64decode(payload.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="image_b64 inválido (no es base64).")

            debug["decoded_len"] = len(img_bytes)

            # detectar MIME si no vino
            if not payload.mime:
                import imghdr
                fmt = imghdr.what(None, h=img_bytes)  # 'jpeg','png','webp',...
                payload.mime = f"image/{fmt}" if fmt else "application/octet-stream"

            # armar data URL para la Responses API
            data_url = f"data:{payload.mime};base64,{payload.image_b64}"
            content.append({"type": "input_image", "image_url": data_url})

        # Llamada a OpenAI Responses API
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )

        return {"model": MODEL, "output": resp.output_text, "debug": debug}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

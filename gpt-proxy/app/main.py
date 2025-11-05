import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Logging visible en Cloud Run
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Clave API de OpenAI
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

# Modelo multimodal (vision)
MODEL = os.getenv("MODEL", "gpt-4.1-mini")
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="2.3")

# Entrada del usuario
class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None
    image_url: str | None = None
    mime: str | None = None  # ejemplo: image/jpeg

# Salida de la API
class InferenceOut(BaseModel):
    model: str
    output: str
    branch: str | None = None  # text / vision-b64 / vision-url

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        branch = "text"
        content = [{"type": "input_text", "text": payload.text}]

        # --- PRIORIDAD: URL si está disponible ---
        if payload.image_url:
            branch = "vision-url"
            content.append({"type": "input_image", "image_url": payload.image_url})
            log.info(f"[VISION-URL] model={MODEL}, url_len={len(payload.image_url)}")

        # --- Si no hay URL, usa base64 ---
        elif payload.image_b64:
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            branch = "vision-b64"
            log.info(f"[VISION-B64] model={MODEL}, mime={mime}, b64_len={len(payload.image_b64)}")
            content.append({"type": "input_image", "image_url": data_url})

        else:
            log.info(f"[TEXT] model={MODEL}")

        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )

        return {"model": MODEL, "output": resp.output_text, "branch": branch}

    except Exception:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

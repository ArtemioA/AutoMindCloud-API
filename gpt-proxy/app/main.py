import os, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

# Puedes alternar entre modelos con visión aquí si 'gpt-4o-mini' no tiene visión en tu cuenta:
MODEL = os.getenv("MODEL", "gpt-4o-mini")  # prueba también "gpt-4.1-mini"
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="1.2")
log = logging.getLogger("uvicorn.error")

class InferenceIn(BaseModel):
    text: str
    image_b64: str | None = None
    mime: str | None = None  # ej: "image/jpeg" o "image/png"

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        messages_content = [{"type": "text", "text": payload.text}]

        if payload.image_b64:
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            log.info(f"[VISION] mime={mime} b64_len={len(payload.image_b64)}")
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        else:
            log.info("[TEXT] sin imagen")

        chat = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2,
        )
        out = chat.choices[0].message.content
        return {"model": MODEL, "output": out}

    except Exception:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

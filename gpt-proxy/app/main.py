class InferenceOut(BaseModel):
    model: str
    output: str
    image_b64: Optional[str]
    debug: Dict[str, Any]


@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        has_img = bool(payload.image_b64 and payload.image_b64.strip())
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": payload.text}]

        debug = {
            "has_image_b64": has_img,
            "mime": payload.mime,
            "b64_prefix": (payload.image_b64[:20] if has_img else None),
            "approx_bytes": None,
            "decoded_len": None,
        }

        if has_img:
            approx_bytes = int(len(payload.image_b64) * 0.75)
            debug["approx_bytes"] = approx_bytes
            if approx_bytes > MAX_REQ_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Imagen demasiado grande para esta vía (~>32 MiB)."
                )

            try:
                img_bytes = base64.b64decode(payload.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="image_b64 inválido (no es base64).")

            debug["decoded_len"] = len(img_bytes)

            if not payload.mime:
                import imghdr
                fmt = imghdr.what(None, h=img_bytes)
                payload.mime = f"image/{fmt}" if fmt else "application/octet-stream"

            data_url = f"data:{payload.mime};base64,{payload.image_b64}"
            content.append({"type": "input_image", "image_url": data_url})

        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )

        # Devolver tanto la respuesta del modelo como la imagen
        return {
            "model": MODEL,
            "output": resp.output_text,
            "image_b64": payload.image_b64 if has_img else None,
            "debug": debug,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

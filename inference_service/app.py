from functools import lru_cache
from fastapi import FastAPI
from schemas import TextRequest
from model import load_model, predict_text
import threading

app = FastAPI(title="Inference Service")
MODEL = None
LOAD_ERR = None

def _load_in_bg():
    global MODEL, LOAD_ERR
    try:
        MODEL = load_model()
        print("[app] MODEL_LOADED_OK", flush=True)
    except Exception as e:
        import traceback
        LOAD_ERR = e
        print("[app] MODEL_LOAD_FAILED:", repr(e), flush=True)
        traceback.print_exc()

@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_load_in_bg, daemon=True)
    t.start()
    print("[app] STARTUP_OK (loading in background)", flush=True)

@lru_cache(maxsize=10000)
def cached_predict(text: str) -> float:
    if MODEL is None:
        raise RuntimeError("model_not_loaded")
    return predict_text(MODEL, text)

@app.get("/health")
def health():
    return {"model_loaded": MODEL is not None,
            "load_error": None if LOAD_ERR is None else repr(LOAD_ERR)}

@app.post("/predict")
def predict(req: TextRequest):
    if MODEL is None:
        return {"error": "model not loaded"}, 503
    prob = cached_predict(req.text)
    return {"is_bot_probability": prob}

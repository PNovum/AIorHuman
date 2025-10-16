import os
import uuid
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from schemas import IncomingMessage, Prediction
from config import SERVICE_HOST, SERVICE_PORT
from security import protect_docs

logger = logging.getLogger(__name__)

INFERENCE_HOST = os.getenv("INFERENCE_HOST", "inference_service")
INFERENCE_PORT = int(os.getenv("INFERENCE_SERVICE_PORT_FASTAPI", "8000"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Start up application on {SERVICE_HOST}:{SERVICE_PORT}")
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="youarebot",
    description="Dialog classification gateway (calls inference_service)",
    lifespan=lifespan,
)

app.middleware("http")(protect_docs)

@app.post("/predict", response_model=Prediction)
def predict(message: IncomingMessage) -> Prediction:
    """
    Принимаем сообщение, отправляем текст в inference_service (/predict),
    получаем {"is_bot_probability": float}, собираем Prediction.
    """
    url = f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/predict"
    payload = {"text": message.text}

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            prob = float(r.json()["is_bot_probability"])
    except Exception:
        logger.exception("inference_service call failed")
        prob = 0.0

    return Prediction(
        id=uuid.uuid4(),
        message_id=message.id,
        dialog_id=message.dialog_id,
        participant_index=message.participant_index,
        is_bot_probability=prob,
    )

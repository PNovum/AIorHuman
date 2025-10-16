# ruff: noqa
import os
import json
from pathlib import Path
import mlflow
import pandas as pd

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://158.160.154.25:5050")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://158.160.154.25:9000"))
os.environ.setdefault("AWS_ACCESS_KEY_ID",      os.getenv("AWS_ACCESS_KEY_ID",      "myuser"))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",  os.getenv("AWS_SECRET_ACCESS_KEY",  "mypassword"))

MODEL_NAME = os.getenv("MODEL_NAME", "model-logreg")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

SAMPLE_PATH = Path(__file__).parent / "sample_dialog.json"

def main():
    assert SAMPLE_PATH.exists(), f"no sample file: {SAMPLE_PATH}"

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    dialog = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    _, messages = next(iter(dialog.items()))
    messages = sorted(messages, key=lambda m: int(m.get("message", 0)))

    for m in messages:
        text = str(m.get("text", ""))
        pi = int(m.get("participant_index", 0))
        df = pd.DataFrame({"text_concat": [text]})
        proba = float(getattr(model, "predict_proba", model.predict)(df)[0][-1]) \
                if hasattr(model, "predict_proba") else float(model.predict(df)[0])
        print(f"{pi}: {text}")
        print(f"is_bot_probability: {proba:.6f}")
    model = mlflow.pyfunc.load_model("models:/model-logreg@champion")
    print("OK:", type(model))
if __name__ == "__main__":
    main()

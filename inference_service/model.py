import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

CANDIDATE_ALIASES = ("champion", "production", "challenger")


def _mk_client() -> MlflowClient:
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registry = os.getenv("MLFLOW_REGISTRY_URI", tracking)
    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(registry)

    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    pwd = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user and pwd:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
    return MlflowClient()


def _get_mv_by_alias(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def _last_version(client: MlflowClient, model_name: str):
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _prod_version(client: MlflowClient, model_name: str):
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    prod = [v for v in versions if getattr(v, "current_stage", "") == "Production"]
    return max(prod, key=lambda v: int(v.version)) if prod else None


def _ensure_champion(client: MlflowClient, model_name: str, version: str | int):
    auto = os.getenv("AUTO_SET_CHAMPION", "true").lower() in ("1", "true", "yes", "on")
    if not auto:
        return
    try:
        if _get_mv_by_alias(client, model_name, "champion") is None:
            client.set_registered_model_alias(model_name, "champion", str(version))
            print(f"[model.py] Set alias champion -> {model_name} v{version}", flush=True)
    except Exception as e:
        print(f"[model.py] WARN: failed to set champion alias: {e}", flush=True)


def _iter_registered_models(client: MlflowClient):

    if not hasattr(client, "search_registered_models"):
        raise RuntimeError(
            "MlflowClient.search_registered_models() недоступен. "
            "Задайте MODEL_NAME в окружении или обновите MLflow."
        )
    page_token = None
    while True:
        resp = client.search_registered_models(filter_string="", page_token=page_token, max_results=100)
        for rm in resp:
            yield rm
        page_token = getattr(resp, "token", None) or getattr(resp, "next_page_token", None)
        if not page_token:
            break


def _resolve_model_name(client: MlflowClient) -> str:
    env_name = os.getenv("MODEL_NAME")
    if env_name:
        return env_name

    champion_candidates: list[tuple[str, int]] = []
    for rm in _iter_registered_models(client):
        mv = _get_mv_by_alias(client, rm.name, "champion")
        if mv is not None:
            try:
                champion_candidates.append((rm.name, int(mv.version)))
            except Exception:
                pass
    if champion_candidates:
        champion_candidates.sort(key=lambda x: x[1], reverse=True)
        return champion_candidates[0][0]

    latest_pool: list[tuple[str, int]] = []
    for rm in _iter_registered_models(client):
        mv = _last_version(client, rm.name)
        if mv is not None:
            try:
                latest_pool.append((rm.name, int(mv.version)))
            except Exception:
                pass
    if not latest_pool:
        raise RuntimeError("No registered models found in MLflow Registry")
    latest_pool.sort(key=lambda x: x[1], reverse=True)
    return latest_pool[0][0]


def _uri_by_alias_or_tag(client: MlflowClient, model_name: str) -> str:
    for alias in CANDIDATE_ALIASES:
        mv = _get_mv_by_alias(client, model_name, alias)
        if mv is not None:
            if alias != "champion":
                _ensure_champion(client, model_name, mv.version)
            return f"models:/{model_name}/{mv.version}"

    mv = _prod_version(client, model_name)
    if mv is not None:
        _ensure_champion(client, model_name, mv.version)
        return f"models:/{model_name}/{mv.version}"

    mv = _last_version(client, model_name)
    if mv is None:
        raise RuntimeError(f"No versions found for registered model '{model_name}'")
    _ensure_champion(client, model_name, mv.version)
    return f"models:/{model_name}/{mv.version}"


def load_model():
    client = _mk_client()
    model_name = _resolve_model_name(client)
    uri = _uri_by_alias_or_tag(client, model_name)
    print(f"[model.py] Loading model: name='{model_name}', uri='{uri}'", flush=True)
    return mlflow.pyfunc.load_model(uri)


def predict_text(model, text: str) -> float:
    df = pd.DataFrame({"text": [text]})
    y = model.predict(df)
    return float(y[0]) if hasattr(y, "__len__") else float(y)

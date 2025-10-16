# ruff: noqa
import io, json, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score

import mlflow, mlflow.sklearn

DATA_DIR   = Path(__file__).parent / "data"
ZIP_PATH   = DATA_DIR / "you-are-bot.zip"
MLFLOW_EXPERIMENT = "week-4"

# ---------- utils ----------
def _load_json(z, name):
    with z.open(name) as f:
        return json.load(io.TextIOWrapper(f, encoding="utf-8"))

def _load_csv(z, name):
    with z.open(name) as f:
        return pd.read_csv(f)

def _json_to_df(d: dict) -> pd.DataFrame:
    rows=[]
    for did, msgs in d.items():
        for m in msgs:
            rows.append({
                "dialog_id": did,
                "participant_index": int(m["participant_index"]),
                "message": int(m.get("message", 0)),
                "text": str(m.get("text","")),
            })
    return pd.DataFrame(rows)

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["dialog_id","participant_index"], as_index=False)["text"]
              .apply(lambda s: " ".join(s.astype(str)))
              .rename(columns={"text":"text_concat"}))

def _apply_consensus_filter(df: pd.DataFrame, num_estimates:int, consent_threshold:float) -> pd.DataFrame:
    est_col = next((c for c in ["num_estimates","n_votes","votes_count"] if c in df.columns), None)
    con_col = next((c for c in ["consent","agreement","consensus","majority_share"] if c in df.columns), None)
    if est_col is None:
        df = df.copy()
        df["_num_estimates_fallback"] = 1
        est_col = "_num_estimates_fallback"
    if con_col is None:
        df = df.copy()
        df["_consent_fallback"] = 1.0
        con_col = "_consent_fallback"
    return df[(df[est_col] >= num_estimates) & (df[con_col] >= consent_threshold)].copy()

def _build_pipeline(model_class:str, tfidf_params:dict, model_params:dict):
    vec = TfidfVectorizer(
        max_features=tfidf_params.get("max_features", 40000),
        ngram_range=tfidf_params.get("ngram_range", (1,2)),
        min_df=tfidf_params.get("min_df", 2),
    )
    if model_class == "logreg":
        mdl = LogisticRegression(
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            class_weight=model_params.get("class_weight", "balanced"),
            solver=model_params.get("solver", "lbfgs"),
        )
    elif model_class == "svc":
        mdl = SVC(
            C=model_params.get("C", 1.0),
            kernel=model_params.get("kernel", "rbf"),
            probability=True,
            class_weight=model_params.get("class_weight", "balanced"),
        )
    else:
        raise ValueError(f"Unknown model_class: {model_class}")
    return Pipeline([("tfidf", vec), ("model", mdl)])

def run_one(num_estimates:int, consent_threshold:float, model_class:str):
    assert ZIP_PATH.exists(), f"нет архива {ZIP_PATH}"
    with zipfile.ZipFile(ZIP_PATH) as z:
        train_json = _load_json(z, "train.json")
        ytrain     = _load_csv (z, "ytrain.csv")
        test_json  = _load_json(z, "test.json")
        ytest      = _load_csv (z, "ytest.csv")

    tr = _json_to_df(train_json)
    tr_agg = _aggregate(tr)
    ytrain["participant_index"] = ytrain["participant_index"].astype(int)
    train_df = tr_agg.merge(ytrain, on=["dialog_id","participant_index"], how="inner")
    train_df = _apply_consensus_filter(train_df, num_estimates, consent_threshold)

    X = train_df["text_concat"].fillna("").values
    y = train_df["is_bot"].astype(int).values
    val_size = max(0.1, min(0.3, 100 / max(100, len(train_df))))
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=val_size, random_state=42,
                                            stratify=y if len(np.unique(y))>1 else None)

    tfidf_params = {"max_features": 40000, "ngram_range": (1,2), "min_df": 2}
    if model_class == "logreg":
        model_params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"}
    else:
        model_params = {"C": 1.0, "kernel": "rbf", "class_weight": "balanced"}

    pipe = _build_pipeline(model_class, tfidf_params, model_params)

    try: mlflow.enable_system_metrics_logging()
    except Exception: pass
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{model_class}-ne{num_estimates}-ct{consent_threshold:.1f}", log_system_metrics=False):
        mlflow.log_param("num_estimates", num_estimates)
        mlflow.log_param("consent_threshold", consent_threshold)
        mlflow.log_param("model_type", model_class)
        mlflow.log_param("model_class", model_class)

        for k,v in model_params.items():
            mlflow.log_param(f"model_params.{k}", v)
        mlflow.log_param("data_processing_params.tfidf_max_features", tfidf_params["max_features"])
        mlflow.log_param("data_processing_params.tfidf_min_df", tfidf_params["min_df"])
        mlflow.log_param("data_processing_params.tfidf_ngram", f"{tfidf_params['ngram_range'][0]}-{tfidf_params['ngram_range'][1]}")

        pipe.fit(Xtr, ytr)

        p_tr = pipe.predict_proba(Xtr)[:,1]
        yhat_tr = (p_tr >= 0.5).astype(int)
        mlflow.log_metric("train_num_dialogs", int(len(Xtr)))
        mlflow.log_metric("train_logloss", float(log_loss(ytr, p_tr, labels=[0,1])))
        mlflow.log_metric("train_accuracy", float(accuracy_score(ytr, yhat_tr)))
        mlflow.log_metric("train_roc_auc", float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else 0.0)
        mlflow.log_metric("train_f1", float(f1_score(ytr, yhat_tr)) if len(np.unique(ytr))>1 else 0.0)

        p_val = pipe.predict_proba(Xval)[:,1]
        yhat_val = (p_val >= 0.5).astype(int)
        mlflow.log_metric("val_num_dialogs", int(len(Xval)))
        mlflow.log_metric("val_logloss", float(log_loss(yval, p_val, labels=[0,1])))
        mlflow.log_metric("val_accuracy", float(accuracy_score(yval, yhat_val)))
        mlflow.log_metric("val_roc_auc", float(roc_auc_score(yval, p_val)) if len(np.unique(yval))>1 else 0.0)
        mlflow.log_metric("val_f1", float(f1_score(yval, yhat_val)) if len(np.unique(yval))>1 else 0.0)

        model_name = f"model-{model_class}"
        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=model_name)

def main():
    combos = []
    for ne in range(1, 6):
        for ct in [0.5,0.6,0.7,0.8,0.9]:
            for mc in ["logreg", "svc"]:
                combos.append((ne, ct, mc))
    for i, (ne, ct, mc) in enumerate(combos[:10], 1):
        print(f"[{i}/10] num_estimates={ne}, consent_threshold={ct}, model_class={mc}")
        run_one(ne, ct, mc)

if __name__ == "__main__":
    main()

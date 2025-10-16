# ruff: noqa
import io, json, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss, accuracy_score, roc_auc_score, f1_score
)
import joblib, mlflow, mlflow.sklearn


DATA_DIR   = Path(__file__).parent / "data"
ZIP_PATH   = DATA_DIR / "you-are-bot.zip"
MODELS_DIR = Path(__file__).parent / "models"; MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_EXPERIMENT = "week-4"

NUM_ESTIMATES      = int(os.getenv("NUM_ESTIMATES", "1"))
CONSENT_THRESHOLD  = float(os.getenv("CONSENT_THRESHOLD", "0.8"))
MODEL_TYPE         = os.getenv("MODEL_TYPE", "logreg")

MODEL_PARAMS = {
    "C": float(os.getenv("LR_C", "1.0")),
    "max_iter": int(os.getenv("LR_MAX_ITER", "1000")),
    "class_weight": os.getenv("LR_CLASS_WEIGHT", "balanced"),
    "solver": os.getenv("LR_SOLVER", "lbfgs"),
}
DATA_PROCESSING_PARAMS = {
    "tfidf_max_features": int(os.getenv("TFIDF_MAX_FEATURES", "40000")),
    "tfidf_min_df": int(os.getenv("TFIDF_MIN_DF", "2")),
    "tfidf_ngram_low": int(os.getenv("TFIDF_NGRAM_LOW", "1")),
    "tfidf_ngram_high": int(os.getenv("TFIDF_NGRAM_HIGH", "2")),
}

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
                "text": str(m.get("text","")),
            })
    return pd.DataFrame(rows)

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["dialog_id","participant_index"], as_index=False)["text"]
              .apply(lambda s: " ".join(s.astype(str)))
              .rename(columns={"text":"text_concat"}))

def _apply_consensus_filter(df: pd.DataFrame) -> pd.DataFrame:
    estimates_col_candidates = ["num_estimates", "n_votes", "votes_count"]
    consent_col_candidates   = ["consent", "agreement", "consensus", "majority_share"]

    est_col = next((c for c in estimates_col_candidates if c in df.columns), None)
    con_col = next((c for c in consent_col_candidates if c in df.columns), None)

    if est_col is None:
        df["_num_estimates_fallback"] = NUM_ESTIMATES if NUM_ESTIMATES > 1 else 1
        est_col = "_num_estimates_fallback"
    if con_col is None:
        df["_consent_fallback"] = 1.0
        con_col = "_consent_fallback"

    return df[(df[est_col] >= NUM_ESTIMATES) & (df[con_col] >= CONSENT_THRESHOLD)].copy()

def main():
    assert ZIP_PATH.exists(), f"нет архива {ZIP_PATH}"
    with zipfile.ZipFile(ZIP_PATH) as z:
        train_json = _load_json(z, "train.json")
        ytrain     = _load_csv (z, "ytrain.csv")
        test_json  = _load_json(z, "test.json")
        ytest      = _load_csv (z, "ytest.csv")

    tr_msgs   = _json_to_df(train_json)
    tr_agg    = _aggregate(tr_msgs)
    ytrain["participant_index"] = ytrain["participant_index"].astype(int)
    train_df = tr_agg.merge(ytrain, on=["dialog_id","participant_index"], how="inner")

    train_df = _apply_consensus_filter(train_df)

    te_msgs  = _json_to_df(test_json)
    te_agg   = _aggregate(te_msgs)
    test_df  = ytest.merge(te_agg, on=["dialog_id","participant_index"], how="left")

    train_df["text_concat"] = train_df["text_concat"].fillna("")
    test_df["text_concat"]  = test_df["text_concat"].fillna("")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=DATA_PROCESSING_PARAMS["tfidf_max_features"],
            ngram_range=(DATA_PROCESSING_PARAMS["tfidf_ngram_low"], DATA_PROCESSING_PARAMS["tfidf_ngram_high"]),
            min_df=DATA_PROCESSING_PARAMS["tfidf_min_df"]
        )),
        ("lr", LogisticRegression(
            C=MODEL_PARAMS["C"],
            max_iter=MODEL_PARAMS["max_iter"],
            class_weight=MODEL_PARAMS["class_weight"],
            solver=MODEL_PARAMS["solver"],
        )),
    ])

    X = train_df["text_concat"].values
    y = train_df["is_bot"].astype(int).values

    val_size = max(0.1, min(0.3, 100 / max(100, len(train_df))))
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    try:
        mlflow.enable_system_metrics_logging()
    except Exception:
        pass

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="tfidf_logreg"):
        mlflow.log_param("num_estimates", NUM_ESTIMATES)
        mlflow.log_param("consent_threshold", CONSENT_THRESHOLD)
        mlflow.log_param("model_type", MODEL_TYPE)

        mlflow.log_param("model_params.C", MODEL_PARAMS["C"])
        mlflow.log_param("model_params.max_iter", MODEL_PARAMS["max_iter"])
        mlflow.log_param("model_params.class_weight", MODEL_PARAMS["class_weight"])
        mlflow.log_param("model_params.solver", MODEL_PARAMS["solver"])


        mlflow.log_param("data_processing_params.tfidf_max_features", DATA_PROCESSING_PARAMS["tfidf_max_features"])
        mlflow.log_param("data_processing_params.tfidf_min_df", DATA_PROCESSING_PARAMS["tfidf_min_df"])
        mlflow.log_param("data_processing_params.tfidf_ngram", f"{DATA_PROCESSING_PARAMS['tfidf_ngram_low']}-{DATA_PROCESSING_PARAMS['tfidf_ngram_high']}")

        pipe.fit(Xtr, ytr)

        # метрики train 
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

        pipe.fit(X, y)
        proba = pipe.predict_proba(test_df["text_concat"].values)[:,1]
        submit = pd.DataFrame({"ID": test_df["ID"].values, "is_bot": proba})
        out_csv = DATA_DIR / "submission.csv"
        submit.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))

        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name="model")
        joblib.dump(pipe, MODELS_DIR / "model.pkl")

    print("OK →", out_csv)

if __name__ == "__main__":
    main()

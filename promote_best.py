# ruff: noqa
import mlflow
from mlflow import MlflowClient


EXPERIMENT = "week-4"
MODEL_CLASSES = ["logreg", "svc"]

def pick_and_promote():
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    assert exp is not None, f"Experiment {EXPERIMENT} not found"
    exp_id = exp.experiment_id

    for mc in MODEL_CLASSES:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"params.model_class = '{mc}'",
            order_by=["metrics.val_logloss ASC"],
            max_results=20,
        )
        if len(runs) < 2:
            print(f"[WARN] Not enough runs for {mc} to assign aliases")
            continue

        best = runs[0]
        second = runs[1]
        print(f"[{mc}] best run_id={best.info.run_id}, val_logloss={best.data.metrics.get('val_logloss')}")
        print(f"[{mc}] second run_id={second.info.run_id}, val_logloss={second.data.metrics.get('val_logloss')}")

        model_name = f"model-{mc}"
        versions = client.search_model_versions(f"name='{model_name}'")
        def find_version_for_run(rid):
            for v in versions:
                if v.run_id == rid:
                    return v.version
            return None

        v_best = find_version_for_run(best.info.run_id)
        v_second = find_version_for_run(second.info.run_id)
        assert v_best and v_second, "Cannot find model versions for runs"

        client.set_registered_model_alias(model_name, "champion", v_best)
        client.set_registered_model_alias(model_name, "challenger", v_second)
        print(f"[{mc}] aliases set: champion -> v{v_best}, challenger -> v{v_second}")

if __name__ == "__main__":
    pick_and_promote()

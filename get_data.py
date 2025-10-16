# ruff: noqa: F401
import argparse
import logging
import pathlib
import time
import pandas as pd
from io import BytesIO


LOGGER: logging.Logger = logging.getLogger("Homework File Downloader")


def _ensure_dir(save_dir: pathlib.Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, save_dir: pathlib.Path, name: str) -> None:
    out = save_dir / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    LOGGER.info("Saved %s (%d rows) -> %s", name, len(df), out)


def get_data_from_minio(save_dir: pathlib.Path) -> None:
    """Download dataset from Minio."""
    LOGGER.info("Downloading dataset from Minio...")

    import boto3  # OR from minio import Minio

    _ensure_dir(save_dir)

    s3 = boto3.client(
        "s3",
        endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id="myuser",
        aws_secret_access_key="mypassword",
    )
    bucket = "mybucket"

    def _get_csv(key: str) -> pd.DataFrame:
        obj = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(obj))

    def _get_json(key: str) -> pd.DataFrame:
        obj = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            return pd.read_json(BytesIO(obj), lines=True)
        except ValueError:
            return pd.read_json(BytesIO(obj))

    train = _get_json("train.json")
    test = _get_json("test.json")
    ytrain = _get_csv("ytrain.csv")
    ytest = _get_csv("ytest.csv")
    sample = _get_csv("sample_submission.csv")

    _save(train, save_dir, "train")
    _save(test, save_dir, "test")
    _save(ytrain, save_dir, "ytrain")
    _save(ytest, save_dir, "ytest")
    _save(sample, save_dir, "sample_submission")


def get_data_from_postgres(save_dir: pathlib.Path) -> None:
    """Download dataset from PostgreSQL."""
    LOGGER.info("Downloading dataset from Postgres...")

    import psycopg2

    conn_params = {
        "host": "localhost",
        "port": 5432,
        "dbname": "mydb",
        "user": "myuser",
        "password": "mypassword",
    }

    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="mydb",
        user="myuser",
        password="mypassword",
    )
    def _q(tab: str) -> pd.DataFrame:
        return pd.read_sql_query(f'SELECT * FROM "{tab}"', conn)

    train = _q("train")
    test = _q("test")
    ytrain = _q("ytrain")
    ytest = _q("ytest")
    sample = _q("sample_submission")

    _save(train, save_dir, "train")
    _save(test, save_dir, "test")
    _save(ytrain, save_dir, "ytrain")
    _save(ytest, save_dir, "ytest")
    _save(sample, save_dir, "sample_submission")


def get_data_from_clickhouse(save_dir: pathlib.Path) -> None:
    """Download dataset from Clickhouse."""
    LOGGER.info("Downloading dataset from Clickhouse...")

    from clickhouse_driver import Client

    _ensure_dir(save_dir)

    client = Client(host="127.0.0.1", port=9002)
    def _q(tab: str) -> pd.DataFrame:
        data = client.execute(f"SELECT * FROM {tab}")
        cols = [c[0] for c in client.execute(f"DESCRIBE TABLE {tab}")]
        return pd.DataFrame(data, columns=cols)

    train = _q("train")
    test = _q("test")
    ytrain = _q("ytrain")
    ytest = _q("ytest")
    sample = _q("sample_submission")

    _save(train, save_dir, "train")
    _save(test, save_dir, "test")
    _save(ytrain, save_dir, "ytrain")
    _save(ytest, save_dir, "ytest")
    _save(sample, save_dir, "sample_submission")


def get_data_from_mongo(save_dir: pathlib.Path) -> None:
    """Download dataset from MongoDB."""
    LOGGER.info("Downloading dataset from MongoDB...")

    from pymongo import MongoClient

    _ensure_dir(save_dir)

    uri = "mongodb://myuser:mypassword@127.0.0.1:27017/?authSource=admin"
    client = MongoClient(uri)
    db = client["mydb"]

    def _c(name: str) -> pd.DataFrame:
        docs = list(db[name].find({}, {"_id": 0}))
        return pd.DataFrame(docs)

    train = _c("train")
    test = _c("test")
    ytrain = _c("ytrain")
    ytest = _c("ytest")
    sample = _c("sample_submission")

    _save(train, save_dir, "train")
    _save(test, save_dir, "test")
    _save(ytrain, save_dir, "ytrain")
    _save(ytest, save_dir, "ytest")
    _save(sample, save_dir, "sample_submission")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    data_folder_path = pathlib.Path(__file__).parents[1] / "data"
    parser = argparse.ArgumentParser(description="Download data for analysis.")
    parser.add_argument(
        "--source",
        choices=["minio", "postgres", "clickhouse", "mongo"],
        default="minio",
        help="Specify the source to download data from.",
    )
    parser.add_argument(
        "--save-dir",
        default=data_folder_path,
        type=pathlib.Path,
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    {
        "minio": get_data_from_minio,
        "postgres": get_data_from_postgres,
        "clickhouse": get_data_from_clickhouse,
        "mongo": get_data_from_mongo,
    }[args.source](args.save_dir)
    LOGGER.info(
        f"Done with `{args.source}` in {time.perf_counter() - start_time:.4f} sec"
    )

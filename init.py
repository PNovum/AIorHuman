import argparse
import io
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
from clickhouse_driver import Client
from minio import Minio
from pymongo import MongoClient
from sqlalchemy import create_engine


def upload_to_minio(data: bytes, filename="you-are-bot.zip"):
    """Uploads data to MinIO."""
    print(f"Start uploading to MinIO")
    try:
        client = Minio(
            "minio:9000", access_key="myuser", secret_key="mypassword", secure=False
        )
        found = client.bucket_exists("mybucket")
        if not found:
            client.make_bucket("mybucket")

        client.put_object("mybucket", filename, io.BytesIO(data), len(data))
        print(f"Data uploaded to MinIO successfully in 'mybucket/{filename}'.")
    except Exception as e:
        print(f"Error uploading to MinIO: {e}")


def upload_to_postgres(data: pd.DataFrame, table_name="train"):
    """Uploads CSV data to PostgreSQL."""
    print(f"Start uploading to PostgreSQL: {table_name}")
    try:
        engine = create_engine("postgresql://myuser:mypassword@postgres:5432/mydb")
        data.to_sql(
            table_name, engine, if_exists="replace", index=False, chunksize=10000
        )
        print(f"Data uploaded to PostgreSQL successfully in table '{table_name}'.")
    except Exception as e:
        print(f"Error uploading to PostgreSQL: {e}")


def upload_to_clickhouse(data: pd.DataFrame, table_name="train"):
    """Uploads CSV data to ClickHouse."""
    print(f"Start uploading to Clickhouse: {table_name}")
    try:
        client = Client(host="clickhouse", port=9000, settings={"use_numpy": True})
        numpy_to_clickhouse_types = {
            np.dtype("O"): "String",
            np.dtype("int64"): "UInt64",
            np.dtype("float64"): "Float64",
        }
        columns = ", ".join(
            f'`{col_name}` "{numpy_to_clickhouse_types[col_type]}"'
            for col_name, col_type in data.dtypes.to_dict().items()
        )
        order_by = ",".join(
            f"`{col}`"
            for col in {"dialog_id", "message", "ID"}.intersection(data.columns)
        )
        client.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} ({columns}) 
            ENGINE = MergeTree() ORDER BY ({order_by})
        """)
        client.insert_dataframe(f"INSERT INTO {table_name} VALUES", data)
        print(f"Data uploaded to ClickHouse successfully in table '{table_name}'.")
    except Exception as e:
        print(f"Error uploading to ClickHouse: {e}")


def upload_to_mongodb(data: pd.DataFrame, collection_name="train"):
    """Uploads CSV data to MongoDB."""
    print(f"Start uploading to MongoDB: {collection_name}")
    try:
        client = MongoClient("mongodb://myuser:mypassword@mongo:27017/")
        db = client["mydb"]
        collection = db[collection_name]
        if collection_name in db.list_collection_names():
            collection.drop()
        collection.insert_many(data.to_dict("records"))
        print(
            f"Data uploaded to MongoDB successfully in collection '{collection_name}'."
        )
    except Exception as e:
        print(f"Error uploading to MongoDB: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload CSV data to various databases."
    )
    parser.add_argument(
        "--target",
        choices=["minio", "postgres", "clickhouse", "mongo", "all"],
        default="all",
        help="Specify the target to upload to. Defaults to 'all'.",
    )
    args = parser.parse_args()

    archive = zipfile.ZipFile(pathlib.Path("data") / "you-are-bot.zip")

    labels_train = pd.read_csv(io.BytesIO(archive.open("ytrain.csv").read()))
    labels_test = pd.read_csv(io.BytesIO(archive.open("ytest.csv").read()))
    sample_submission = pd.read_csv(
        io.BytesIO(archive.open("sample_submission.csv").read())
    )
    train = pd.DataFrame(
        [
            {"dialog_id": dialog_id, **prop}
            for dialog_id, props in json.loads(
                archive.open("train.json").read()
            ).items()
            for prop in props
        ]
    )
    test = pd.DataFrame(
        [
            {"dialog_id": dialog_id, **prop}
            for dialog_id, props in json.loads(archive.open("test.json").read()).items()
            for prop in props
        ]
    )

    def _upload(func):
        func(train, "train")
        func(test, "test")
        func(labels_train, "ytrain")
        func(labels_test, "ytest")
        func(sample_submission, "sample_submission")

    if args.target == "all" or args.target == "minio":
        for file_name in archive.filelist:
            upload_to_minio(archive.open(file_name).read(), filename=file_name.filename)
    if args.target == "all" or args.target == "postgres":
        _upload(upload_to_postgres)
    if args.target == "all" or args.target == "clickhouse":
        _upload(upload_to_clickhouse)
    if args.target == "all" or args.target == "mongo":
        _upload(upload_to_mongodb)

    print("Done!")


if __name__ == "__main__":
    main()

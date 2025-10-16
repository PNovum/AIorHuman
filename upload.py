# ruff: noqa: F401
import argparse
import logging
import pathlib
import os

LOGGER: logging.Logger = logging.getLogger("Homework S3 Upload Stage")


def upload_to_s3(file_path: pathlib.Path):
    import boto3

    bucket = os.environ["MINIO_BUCKET_NAME"]
    port = os.environ["MINIO_PORT"]
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]

    object_name = pathlib.Path(file_path).name

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{port}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    s3.upload_file(str(file_path), bucket, object_name)
    print(f"Uploaded to s3://{bucket}/{object_name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    data_folder_path = pathlib.Path(__file__).parents[1] / "data"
    parser = argparse.ArgumentParser(description="Analyze data")
    parser.add_argument(
        "--file-path",
        default=data_folder_path / "analysis.parquet",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    upload_to_s3(args.file_path)
    LOGGER.info("Done")
# ruff: noqa: F401
import argparse
import logging
import pathlib
import time

LOGGER: logging.Logger = logging.getLogger("Homework Analysis Stage")


def analyze_data_pandas(
    train_path: pathlib.Path, ytrain_path: pathlib.Path, results_path: pathlib.Path
) -> None:
    """Analyze data with pandas and save results to `results_path`."""
    import pandas as pd

    train = pd.read_parquet(train_path)
    ytrain = pd.read_parquet(ytrain_path)

    train_with_labels = pd.merge(
        train.astype({"participant_index": int}),
        ytrain,
        on=["dialog_id", "participant_index"],
    )

    result = (
        train_with_labels.groupby("dialog_id")
        .agg(
            num_chars=("text", lambda x: x.str.len().sum()),
            mean_chars=("text", lambda x: x.str.len().mean()),
            mean_punct=("text", lambda x: x.str.contains(r"[,.]").mean()),
            num_messages=("text", "count"),
            has_bot=("is_bot", "max"),
        )
        .query("num_messages > 5")
        .pipe(
            lambda df: pd.DataFrame(
                {
                    "num_dialogs": [len(df)],
                    "num_bot_dialogs": [df["has_bot"].sum()],
                    "num_human_dialogs": [(~df["has_bot"].astype(bool)).sum()],
                    "avg_dialog_length": [df["num_chars"].mean()],
                    "avg_text_length": [df["mean_chars"].mean()],
                    "avg_num_messages": [df["num_messages"].mean()],
                    "punctuation_is_bot_corr": [df["mean_punct"].corr(df["has_bot"])],
                }
            )
        )
    )

    result.to_parquet(results_path)


def analyze_data_polars(
    train_path: pathlib.Path, ytrain_path: pathlib.Path, results_path: pathlib.Path
) -> None:
    import polars as pl

    train = pl.scan_parquet(str(train_path))
    ytrain = pl.scan_parquet(str(ytrain_path))

    train = train.with_columns(pl.col("participant_index").cast(pl.Int64))

    joined = train.join(
        ytrain, on=["dialog_id", "participant_index"], how="inner"
    )

    per_dialog = (
        joined
        .group_by("dialog_id")
        .agg(
            num_chars = pl.col("text").str.len_chars().sum(),
            mean_chars = pl.col("text").str.len_chars().mean(),
            mean_punct = pl.col("text").str.contains(r"[,.]").cast(pl.Float64).mean(),
            num_messages = pl.count(),
            has_bot = pl.col("is_bot").max().cast(pl.Int64),
        )
        .filter(pl.col("num_messages") > 5)
    )

    result = per_dialog.select(
        pl.len().alias("num_dialogs"),
        pl.col("has_bot").sum().alias("num_bot_dialogs"),
        (pl.col("has_bot") == 0).sum().alias("num_human_dialogs"),
        pl.col("num_chars").mean().alias("avg_dialog_length"),
        pl.col("mean_chars").mean().alias("avg_text_length"),
        pl.col("num_messages").mean().alias("avg_num_messages"),
        pl.corr(pl.col("mean_punct"), pl.col("has_bot").cast(pl.Float64)).alias("punctuation_is_bot_corr"),
    ).collect()

    result.select([
        "num_dialogs",
        "num_bot_dialogs",
        "num_human_dialogs",
        "avg_dialog_length",
        "avg_text_length",
        "avg_num_messages",
        "punctuation_is_bot_corr",
    ]).write_parquet(str(results_path))


def analyze_data_duckdb(
    train_path: pathlib.Path, ytrain_path: pathlib.Path, results_path: pathlib.Path
) -> None:
    import duckdb
    import pandas as pd

    con = duckdb.connect()
    # читаем parquet напрямую с диска во временные таблицы
    con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM read_parquet(?);", [str(train_path)])
    con.execute("CREATE OR REPLACE TABLE ytrain AS SELECT * FROM read_parquet(?);", [str(ytrain_path)])
    con.execute("UPDATE train SET participant_index = CAST(participant_index AS BIGINT);")

    df: pd.DataFrame = con.execute(
        """
        WITH joined AS (
            SELECT t.dialog_id, t.participant_index, t.text, y.is_bot
            FROM train t
            JOIN ytrain y USING (dialog_id, participant_index)
        ),
        per_dialog AS (
            SELECT
                dialog_id,
                SUM(length(text))                                        AS num_chars,
                AVG(length(text))                                        AS mean_chars,
                AVG(CASE WHEN regexp_matches(text, '[,.]') THEN 1 ELSE 0 END) :: DOUBLE AS mean_punct,
                COUNT(*)                                                 AS num_messages,
                MAX(is_bot)                                              AS has_bot
            FROM joined
            GROUP BY dialog_id
        ),
        filtered AS (
            SELECT * FROM per_dialog WHERE num_messages > 5
        )
        SELECT
            COUNT(*)                                                   AS num_dialogs,
            SUM(has_bot)                                               AS num_bot_dialogs,
            SUM(CASE WHEN has_bot = 0 THEN 1 ELSE 0 END)               AS num_human_dialogs,
            AVG(num_chars)                                             AS avg_dialog_length,
            AVG(mean_chars)                                            AS avg_text_length,
            AVG(num_messages)                                          AS avg_num_messages,
            corr(mean_punct, CAST(has_bot AS DOUBLE))                  AS punctuation_is_bot_corr
        FROM filtered;
        """
    ).fetch_df()

    df = df[
        [
            "num_dialogs",
            "num_bot_dialogs",
            "num_human_dialogs",
            "avg_dialog_length",
            "avg_text_length",
            "avg_num_messages",
            "punctuation_is_bot_corr",
        ]
    ]
    df.to_parquet(results_path, index=False)

def analyze_data_fireducks(
    train_path: pathlib.Path, ytrain_path: pathlib.Path, results_path: pathlib.Path
) -> None:
    import fireducks.pandas as pd

    train = pd.read_parquet(train_path)
    ytrain = pd.read_parquet(ytrain_path)

    train_with_labels = pd.merge(
        train.astype({"participant_index": int}),
        ytrain,
        on=["dialog_id", "participant_index"],
    )

    result = (
        train_with_labels.groupby("dialog_id")
        .agg(
            num_chars=("text", lambda x: x.str.len().sum()),
            mean_chars=("text", lambda x: x.str.len().mean()),
            mean_punct=("text", lambda x: x.str.contains(r"[,.]").mean()),
            num_messages=("text", "count"),
            has_bot=("is_bot", "max"),
        )
        .query("num_messages > 5")
        .pipe(
            lambda df: pd.DataFrame(
                {
                    "num_dialogs": [len(df)],
                    "num_bot_dialogs": [df["has_bot"].sum()],
                    "num_human_dialogs": [(~df["has_bot"].astype(bool)).sum()],
                    "avg_dialog_length": [df["num_chars"].mean()],
                    "avg_text_length": [df["mean_chars"].mean()],
                    "avg_num_messages": [df["num_messages"].mean()],
                    "punctuation_is_bot_corr": [df["mean_punct"].corr(df["has_bot"])],
                }
            )
        )
    )

    result.to_parquet(results_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
    data_folder_path = pathlib.Path(__file__).parents[1] / "data"
    parser = argparse.ArgumentParser(description="Analyze data")
    parser.add_argument(
        "--train-path",
        default=data_folder_path / "train.parquet",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--ytrain-path",
        default=data_folder_path / "ytrain.parquet",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--save-path",
        default=data_folder_path / "analysis.parquet",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--implementation",
        choices=["pandas", "polars", "duckdb", "fireducks"],
        default="pandas",
        help="Specify the analysis implementation type.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    {
        "pandas": analyze_data_pandas,
        "polars": analyze_data_polars,
        "duckdb": analyze_data_duckdb,
        "fireducks": analyze_data_fireducks,
    }[args.implementation](args.train_path, args.ytrain_path, args.save_path)
    LOGGER.info(
        f"Analysis done with `{args.implementation}` in {time.perf_counter() - start_time:.4f} sec"
    )

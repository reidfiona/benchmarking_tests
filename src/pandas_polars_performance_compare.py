import statistics
import timeit
import csv
import pandas as pd
import numpy as np
import polars as pl


def measure_performance(func, n_runs=50):
    times = timeit.repeat(func, repeat=n_runs, number=1)
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times)
    retval = func()
    return retval, mean_time, median_time, std_time


def grp_agg_pandas(df: pd.DataFrame):
    return (
        df.groupby("user_id")
        .agg(
            num_actions=("action_type", "count"),
            avg_session_duration=("session_duration", "mean"),
        )
        .reset_index()
    )


def p90_pandas(df: pd.DataFrame):
    return df["num_actions"].quantile(0.9)


def filter_pandas(df: pd.DataFrame, top_10_percent_threshold: float):
    return df[df["num_actions"] >= top_10_percent_threshold]


def sort_pandas(df, sort_by: str):
    return df.sort_values(sort_by, ascending=False)


def grp_agg_polars(df: pl.DataFrame):
    return df.group_by("user_id").agg(
        [
            pl.count("action_type").alias("num_actions"),
            pl.col("session_duration").mean().alias("avg_session_duration"),
        ]
    )


def p90_polars(df: pl.DataFrame):
    return df.select(
        [pl.quantile("num_actions", 0.90).alias("top_10_percent_threshold")]
    ).to_series()[0]


def filter_polars(df: pl.DataFrame, top_10_percent_threshold: float):
    return df.filter(pl.col("num_actions") >= top_10_percent_threshold)


def sort_polars(df: pl.DataFrame, sort_by: str, multithreaded=True):
    return df.sort(sort_by, descending=True, multithreaded=multithreaded)


if __name__ == "__main__":
    # Generate dataset
    num_records = 1000000
    num_users = 100000
    num_sessions = 10000
    np.random.seed(42)
    user_ids = np.random.choice(range(1, num_users + 1), num_records)
    action_types = np.random.choice(
        ["click", "view", "purchase"], num_records, p=[0.6, 0.3, 0.1]
    )
    timestamps = pd.date_range(start="2020-01-01", periods=num_records, freq="s")
    session_ids = np.random.randint(1, num_sessions, num_records)
    session_durations = np.random.lognormal(mean=6, sigma=0.75, size=num_records)

    # Create pandas DataFrame
    data = {
        "user_id": user_ids,
        "action_type": action_types,
        "timestamp": timestamps,
        "session_id": session_ids,
        "session_duration": session_durations,
    }
    df = pd.DataFrame(data)

    with open("performance_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Library", "Operation", "Mean Time [s]", "Median Time [s]", "Standard Deviation [s]"]
        )

        user_activity, pandas_mean, pandas_median, pandas_steddev = measure_performance(
            lambda: grp_agg_pandas(df)
        )
        writer.writerow(
            ["pandas", "Group By and Aggregate", pandas_mean, pandas_median, pandas_steddev]
        )

        p90_thresh, pandas_mean, pandas_median, pandas_steddev = measure_performance(
            lambda: p90_pandas(user_activity)
        )
        writer.writerow(
            ["pandas", "Quantile", pandas_mean, pandas_median, pandas_steddev]
        )

        top10_users, pandas_mean, pandas_median, pandas_steddev = measure_performance(
            lambda: filter_pandas(user_activity, p90_thresh)
        )
        writer.writerow(
            ["pandas", "Filter", pandas_mean, pandas_median, pandas_steddev]
        )

        sorted_df, pandas_mean, pandas_median, pandas_steddev = measure_performance(
            lambda: sort_pandas(top10_users, "avg_session_duration")
        )
        writer.writerow(["pandas", "Sort", pandas_mean, pandas_median, pandas_steddev])

        # Convert pandas DataFrame to Polars DataFrame
        df_pl = pl.from_pandas(df)

        user_activity, polars_mean, polars_median, polars_steddev = measure_performance(
            lambda: grp_agg_polars(df_pl)
        )
        writer.writerow(
            ["Polars", "Group By and Aggregate", polars_mean, polars_median, polars_steddev]
        )

        p90_thresh, polars_mean, polars_median, polars_steddev = measure_performance(
            lambda: p90_polars(user_activity)
        )
        writer.writerow(
            ["Polars", "Quantile", polars_mean, polars_median, polars_steddev]
        )

        top10_users, polars_mean, polars_median, polars_steddev = measure_performance(
            lambda: filter_polars(user_activity, p90_thresh)
        )
        writer.writerow(
            ["Polars", "Filter", polars_mean, polars_median, polars_steddev]
        )

        sorted_df, polars_mean, polars_median, polars_steddev = measure_performance(
            lambda: sort_polars(top10_users, "avg_session_duration")
        )
        writer.writerow(
            ["Polars", "Sort - Multithreaded", polars_mean, polars_median, polars_steddev]
        )
        sorted_df, polars_mean, polars_median, polars_steddev = measure_performance(
            lambda: sort_polars(
                top10_users, "avg_session_duration", multithreaded=False
            )
        )
        writer.writerow(
            ["Polars", "Sort - Singlethreaded", polars_mean, polars_median,polars_steddev]
        )
# Running this on my M1 mac gives:
#
#                Performance Comparison: Pandas vs Polars                
# ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
# ┃   Operation    ┃ Pandas Time (s) ┃ Polars Time (s) ┃ Polar Speedup  ┃
# ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
# │    Read CSV    │     39.4148     │     4.9128      │  8.02x faster  │
# │ Select Columns │     0.1412      │     0.0007      │ 194.89x faster │
# │     Filter     │     0.6003      │     0.5218      │  1.15x faster  │
# │      Sort      │     4.5022      │     3.2107      │  1.4x faster   │
# │    Group By    │      0.383      │     1.6509      │  0.23x slower  │
# └────────────────┴─────────────────┴─────────────────┴────────────────┘
#
# Eager vs Lazy API Usage
#        in Polars       
# ┏━━━━━━━━━━━┳━━━━━━━━━┓
# ┃ API Type  ┃ Time(s) ┃
# ┡━━━━━━━━━━━╇━━━━━━━━━┩
# │ Eager API │ 6.6829  │
# │ Lazy API  │ 0.8571  │
# └───────────┴─────────┘
#
# Make sure to download `US_Accidents_March23.csv` from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
# to test it out with this.

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table


@contextmanager
def measure_time() -> Iterator[Callable[[], float]]:
    start_time = end_time = time.perf_counter()
    yield lambda: end_time - start_time
    end_time = time.perf_counter()


csv_path = "../archive/US_Accidents_March23.csv"

with measure_time() as pandas_read_csv:
    pd_df = pd.read_csv(csv_path)

with measure_time() as pandas_select:
    pd_df_selected = pd_df[
        ["Severity", "Start_Time", "End_Time", "Station", "Stop", "Traffic_Signal"]
    ]

with measure_time() as pandas_filter:
    filter_pd_df = pd_df[pd_df["Traffic_Signal"]]

with measure_time() as pandas_sort:
    sorted_pd_df = pd_df.sort_values(by="Humidity(%)", ascending=False)

with measure_time() as pandas_group_by:
    grouped_pd_df = pd_df.groupby(["State"])["ID"].agg("count")

with measure_time() as polar_read_csv:
    pl_df = pl.read_csv(csv_path)

with measure_time() as polar_select:
    pl_df_selected = pl_df[
        ["Severity", "Start_Time", "End_Time", "Station", "Stop", "Traffic_Signal"]
    ]

with measure_time() as polar_filter:
    filter_pl_df = pl_df.filter(pl.col("Traffic_Signal"))

with measure_time() as polar_sort:
    sorted_pl_df = pl_df.sort("Humidity(%)", descending=True)

with measure_time() as polar_group_by:
    grouped_pl_df = pl_df.group_by("State").agg(pl.col("ID").count())

table = Table(title="Performance Comparison: Pandas vs Polars")
table.add_column("Operation", style="cyan", justify="center")
table.add_column("Pandas Time (s)", style="magenta", justify="center")
table.add_column("Polars Time (s)", style="green", justify="center")
table.add_column("Polar Speedup", style="yellow", justify="center")

operations = ["Read CSV", "Select Columns", "Filter", "Sort", "Group By"]

pandas_times = [
    pandas_read_csv(),
    pandas_select(),
    pandas_filter(),
    pandas_sort(),
    pandas_group_by(),
]

polars_times = [polar_read_csv(), polar_select(), polar_filter(), polar_sort(), polar_group_by()]

for operation, pandas_time, polars_time in zip(operations, pandas_times, polars_times):
    speedup = round(pandas_time / polars_time, 2)
    if polars_time < pandas_time:
        speedup = f"[green]{speedup}x faster[/green]"
    elif polars_time > pandas_time:
        speedup = f"[red]{speedup}x slower[/red]"
    else:
        speedup = "Equal"
    table.add_row(operation, str(round(pandas_time, 4)), str(round(polars_time, 4)), speedup)


console = Console()
console.print(table)

api_table = Table(title="Eager vs Lazy API Usage in Polars")
api_table.add_column("API Type", style="cyan", justify="center")
api_table.add_column("Time (s)", style="magenta", justify="center")

with measure_time() as polar_eager_api:
    pl_df = (
        pl.read_csv(csv_path)
        .filter(pl.col("Severity") == 4)
        .group_by(["State", "County"])
        .agg(pl.col("ID").count().alias("Count Severity"))
        .sort("Count Severity", descending=True)
    )

with measure_time() as polar_lazy_api:
    q1 = (
        pl.scan_csv(csv_path)
        .filter(pl.col("Severity") == 4)
        .group_by(["State", "County"])
        .agg(pl.col("ID").count().alias("Count Severity"))
        .sort("Count Severity", descending=True)
        .collect()
    )

api_table.add_row("Eager API", str(round(polar_eager_api(), 4)))
api_table.add_row("Lazy API", str(round(polar_lazy_api(), 4)))

console.print(api_table)

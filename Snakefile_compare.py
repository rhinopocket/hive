"""
Snakemake pipeline — dataset comparison (ref vs cmp)
=====================================================
Compares two hive-partitioned datasets across matched time partitions.

Formats
-------
  Pivoted   : one row per (time_col, product_col), indicators as columns
  Unpivoted : one row per (time_col, product_col, indicator_name, indicator_value)

When formats differ, set align_strategy:
  "pivot"   — unpivot the pivoted side
  "unpivot" — pivot the unpivoted side

Granularity mismatch
--------------------
Detected at DAG-building time. The coarser side is filtered using the
finer partition's exact time values as a serialised Polars expression.

Outputs (per cmp partition, mirroring cmp layout under output_root)
--------------------------------------------------------------------
  keys_only.parquet    : KEY_COLS + source ("ref"|"cmp")
  correlations.parquet : indicator_name, correlation, n_common_keys
  differences.parquet  : KEY_COLS, indicator_name, ref_value, cmp_value,
                         abs_diff, rel_diff

Configuration
-------------
  dataset_ref, dataset_cmp  : paths to dataset roots
  output_root               : where to write results
  time_column               : name of the datetime column (e.g. "bin_time")
  product_column            : name of the product key column (e.g. "prd_code")
  format_ref, format_cmp    : "pivoted" | "unpivoted"
  align_strategy            : "pivot" | "unpivot"  (when formats differ)
  granularity_ref, granularity_cmp : optional YEAR/QUARTER/MONTH/DAY
  checks                    : list of checks to run, subset of:
                                - keys_only
                                - correlations
                                - differences
"""

import sys
from datetime import date as _date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import polars as pl

sys.path.insert(0, workflow.basedir)
from hive_partition_mapper import Dataset, Granularity, PartitionedFile, map_datasets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_REF    = Path(config["dataset_ref"])
DATASET_CMP    = Path(config["dataset_cmp"])
OUTPUT_ROOT    = Path(config["output_root"])
TIME_COL       = config.get("time_column",    "bin_time")
PRODUCT_COL    = config.get("product_column", "prd_code")
FORMAT_REF     = config["format_ref"]
FORMAT_CMP     = config["format_cmp"]
ALIGN          = config.get("align_strategy", None)

VALID_CHECKS   = {"keys_only", "correlations", "differences"}
CHECKS         = config.get("checks", list(VALID_CHECKS))
unknown        = set(CHECKS) - VALID_CHECKS
if unknown:
    raise ValueError(f"Unknown checks: {unknown}. Valid: {VALID_CHECKS}")
if not CHECKS:
    raise ValueError("'checks' is empty — nothing to run.")

if FORMAT_REF != FORMAT_CMP and ALIGN not in ("pivot", "unpivot"):
    raise ValueError(
        f"Formats differ (ref={FORMAT_REF}, cmp={FORMAT_CMP}) — "
        f"set align_strategy to 'pivot' or 'unpivot'."
    )

WORKING_FORMAT = FORMAT_REF if FORMAT_REF == FORMAT_CMP else (
    "pivoted" if ALIGN == "pivot" else "unpivoted"
)

# Key columns depend on working format; use the configured column names
KEY_COLS = (
    [TIME_COL, PRODUCT_COL] if WORKING_FORMAT == "pivoted"
    else [TIME_COL, PRODUCT_COL, "indicator_name"]
)

_GRAN_ORDER = {g: i for i, g in enumerate(Granularity, 1)}  # coarser = smaller int


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def _load(name, root, gran_str):
    gran = Granularity[gran_str.upper()] if gran_str else None
    return (Dataset.from_directory(name, root, time_granularity=gran)
            if gran else Dataset.from_directory(name, root))

ds_ref = _load("ref", DATASET_REF, config.get("granularity_ref", ""))
ds_cmp = _load("cmp", DATASET_CMP, config.get("granularity_cmp", ""))

mappings = map_datasets(ds_cmp, ds_ref)


def _filter_expr_json(finer: PartitionedFile) -> str:
    """Exact Polars filter on TIME_COL for the finer partition's slice."""
    g, s = finer.granularity, finer.start
    if g == Granularity.DAY:
        expr = pl.col(TIME_COL).dt.date() == pl.lit(_date(s.year, s.month, s.day))
    elif g == Granularity.MONTH:
        expr = (pl.col(TIME_COL).dt.year()  == s.year) & \
               (pl.col(TIME_COL).dt.month() == s.month)
    elif g == Granularity.QUARTER:
        q_months = {1: (1,2,3), 4: (4,5,6), 7: (7,8,9), 10: (10,11,12)}
        expr = (pl.col(TIME_COL).dt.year()  == s.year) & \
               (pl.col(TIME_COL).dt.month().is_in(list(q_months[s.month])))
    else:  # YEAR
        expr = pl.col(TIME_COL).dt.year() == s.year
    return expr.meta.serialize(format="json")


# JOBS: output_path → {cmp_path, ref_paths, filter_side, filter_expr}
#
# keys_only must cover ALL cmp partitions, including those with no ref match
# (unmatched → ref_paths=[], all cmp keys will be marked source="cmp").
# correlations and differences only run on matched partitions (no common keys
# possible otherwise), but we register them for all partitions anyway and let
# _compute_* return empty DataFrames for the unmatched ones — this keeps the
# output tree consistent and avoids special-casing in rule all.
JOBS: Dict[str, dict] = {}

# Index matched partitions from map_datasets
matched: Dict[str, dict] = {}
for m in mappings:
    cmp_file  = m.source
    ref_files = m.targets
    if not ref_files:
        filter_side, filter_expr = "none", ""
    else:
        ref_order = _GRAN_ORDER[ref_files[0].granularity]
        cmp_order = _GRAN_ORDER[cmp_file.granularity]
        if ref_order == cmp_order:
            filter_side, filter_expr = "none", ""
        elif ref_order < cmp_order:
            filter_side, filter_expr = "ref", _filter_expr_json(cmp_file)
        else:
            filter_side, filter_expr = "cmp", _filter_expr_json(ref_files[0])
    matched[cmp_file.path] = {
        "ref_paths":   [f.path for f in ref_files],
        "filter_side": filter_side,
        "filter_expr": filter_expr,
    }

# JOBS_CMP: one job per cmp partition (all cmp files, matched or not)
# Output: {OUTPUT_ROOT}/{check}/cmp/{rel}/part.parquet
JOBS_CMP: Dict[str, dict] = {}
for cmp_file in ds_cmp.files:
    rel  = Path(cmp_file.path).relative_to(DATASET_CMP)
    info = matched.get(cmp_file.path, {"ref_paths": [], "filter_side": "none", "filter_expr": ""})
    for check in CHECKS:
        out = str(OUTPUT_ROOT / check / "cmp" / rel / "part.parquet")
        JOBS_CMP[out] = {"cmp_path": cmp_file.path, **info}

# JOBS_REF: one job per unmatched ref partition (keys_only only)
# A ref partition is "unmatched" when no cmp partition overlaps it.
# Output: {OUTPUT_ROOT}/keys_only/ref/{rel}/part.parquet
matched_ref_paths = {rp for info in matched.values() for rp in info["ref_paths"]}
JOBS_REF: Dict[str, dict] = {}
if "keys_only" in CHECKS:
    for ref_file in ds_ref.files:
        if ref_file.path not in matched_ref_paths:
            rel = Path(ref_file.path).relative_to(DATASET_REF)
            out = str(OUTPUT_ROOT / "keys_only" / "ref" / rel / "part.parquet")
            JOBS_REF[out] = {"ref_path": ref_file.path}

JOBS = {**JOBS_CMP, **JOBS_REF}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _read_partition(path: str) -> pl.DataFrame:
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def _apply_filter(df: pl.DataFrame, filter_expr_json: str) -> pl.DataFrame:
    """Apply a serialised Polars filter expression. No-op when empty."""
    if not filter_expr_json or df.is_empty():
        return df
    return df.filter(pl.Expr.deserialize(filter_expr_json.encode(), format="json"))


def _load_filter_align(
    paths: List[str],
    declared_format: str,
    filter_expr_json: str,
) -> pl.DataFrame:
    """
    Read → filter on bin_time → align format.

    The filter is applied while data is still in its native (declared) format,
    before any pivot/unpivot.  This matters for pivoted data: filtering before
    unpivoting avoids exploding rows by the number of indicators only to discard
    most of them immediately after.
    """
    frames = [f for p in paths if not (f := _read_partition(p)).is_empty()]
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames)

    # Filter first, on the compact native representation
    df = _apply_filter(df, filter_expr_json)
    if df.is_empty():
        return df

    # Then align to WORKING_FORMAT
    if declared_format == WORKING_FORMAT:
        return df
    if WORKING_FORMAT == "unpivoted":
        ind_cols = [c for c in df.columns if c not in (TIME_COL, PRODUCT_COL)]
        return df.unpivot(on=ind_cols, index=[TIME_COL, PRODUCT_COL],
                          variable_name="indicator_name", value_name="indicator_value")
    return df.pivot(on="indicator_name", index=[TIME_COL, PRODUCT_COL],
                    values="indicator_value", aggregate_function="first")


def _ind_cols(df: pl.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in (TIME_COL, PRODUCT_COL)]


# ---------------------------------------------------------------------------
# Pre-processing  (shared by all rules)
# ---------------------------------------------------------------------------

def _preprocess(job: dict):
    """
    Load, filter, and align both sides.  Returns (df_ref, df_cmp) ready for
    comparison in WORKING_FORMAT.  Filter is applied before pivot/unpivot.

    JOBS_CMP entries have "cmp_path" + "ref_paths".
    JOBS_REF entries have only "ref_path" (no cmp side — unmatched ref partition).
    """
    if "ref_path" in job:
        # Unmatched ref partition: load ref only, cmp is empty
        df_ref = _load_filter_align([job["ref_path"]], FORMAT_REF, "")
        return df_ref, pl.DataFrame()
    df_cmp = _load_filter_align(
        [job["cmp_path"]], FORMAT_CMP,
        job["filter_expr"] if job["filter_side"] == "cmp" else "",
    )
    df_ref = _load_filter_align(
        job["ref_paths"], FORMAT_REF,
        job["filter_expr"] if job["filter_side"] == "ref" else "",
    )
    return df_ref, df_cmp


# ---------------------------------------------------------------------------
# Processing  (one function per check)
# ---------------------------------------------------------------------------

def _compute_keys_only(df_ref: pl.DataFrame, df_cmp: pl.DataFrame) -> pl.DataFrame:
    key_schema = {k: pl.Utf8 for k in KEY_COLS}
    if df_ref.is_empty() and df_cmp.is_empty():
        return pl.DataFrame(schema={**key_schema, "source": pl.Utf8})
    keys_ref = df_ref.select(KEY_COLS).unique() if not df_ref.is_empty()                else pl.DataFrame(schema=key_schema)
    keys_cmp = df_cmp.select(KEY_COLS).unique() if not df_cmp.is_empty()                else pl.DataFrame(schema=key_schema)
    return pl.concat([
        keys_ref.join(keys_cmp, on=KEY_COLS, how="anti").with_columns(pl.lit("ref").alias("source")),
        keys_cmp.join(keys_ref, on=KEY_COLS, how="anti").with_columns(pl.lit("cmp").alias("source")),
    ])


def _compute_correlations(df_ref: pl.DataFrame, df_cmp: pl.DataFrame) -> pl.DataFrame:
    empty = pl.DataFrame(schema={"indicator_name": pl.Utf8,
                                 "correlation": pl.Float64,
                                 "n_common_keys": pl.Int64})
    if df_ref.is_empty() or df_cmp.is_empty():
        return empty
    joined = df_ref.join(df_cmp, on=KEY_COLS, how="inner", suffix="_cmp")
    if WORKING_FORMAT == "pivoted":
        rows = []
        for ind in _ind_cols(df_ref):
            col_ref, col_cmp = ind, f"{ind}_cmp"
            if col_ref not in joined.columns or col_cmp not in joined.columns:
                continue
            sub  = joined.select([col_ref, col_cmp]).drop_nulls()
            corr = sub.select(pl.corr(col_ref, col_cmp)).item() if len(sub) > 1 else float("nan")
            rows.append({"indicator_name": ind, "correlation": corr, "n_common_keys": len(sub)})
        return pl.from_dicts(rows, schema=empty.schema) if rows else empty
    return joined.group_by("indicator_name").agg([
        pl.corr("indicator_value", "indicator_value_cmp").alias("correlation"),
        pl.len().alias("n_common_keys"),
    ])


def _compute_differences(df_ref: pl.DataFrame, df_cmp: pl.DataFrame) -> pl.DataFrame:
    out_schema = {k: pl.Utf8 for k in KEY_COLS}
    out_schema.update({"indicator_name": pl.Utf8, "ref_value": pl.Float64,
                       "cmp_value": pl.Float64, "abs_diff": pl.Float64,
                       "rel_diff": pl.Float64})
    if df_ref.is_empty() or df_cmp.is_empty():
        return pl.DataFrame(schema=out_schema)
    joined = df_ref.join(df_cmp, on=KEY_COLS, how="inner", suffix="_cmp")
    if WORKING_FORMAT == "pivoted":
        frames = []
        for ind in _ind_cols(df_ref):
            col_ref, col_cmp = ind, f"{ind}_cmp"
            if col_ref not in joined.columns or col_cmp not in joined.columns:
                continue
            frames.append(
                joined.select(KEY_COLS + [col_ref, col_cmp])
                .rename({col_ref: "ref_value", col_cmp: "cmp_value"})
                .with_columns(pl.lit(ind).alias("indicator_name"))
            )
        result = pl.concat(frames) if frames else pl.DataFrame(schema=out_schema)
    else:
        result = (joined
            .select(KEY_COLS + ["indicator_value", "indicator_value_cmp"])
            .rename({"indicator_value": "ref_value", "indicator_value_cmp": "cmp_value"}))
    if not result.is_empty():
        result = result.with_columns([
            (pl.col("ref_value") - pl.col("cmp_value")).abs().alias("abs_diff"),
            ((pl.col("ref_value") - pl.col("cmp_value")).abs()
             / pl.col("ref_value").abs().replace(0, None)).alias("rel_diff"),
        ])
    return result.select(list(out_schema))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

rule all:
    input: list(JOBS)


rule keys_only:
    """
    Keys in ref only or cmp only.
    Covers all cmp partitions (matched or not) under keys_only/cmp/.
    """
    input:
        cmp = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "keys_only" / "cmp" / wc.rel_path / "part.parquet")]["cmp_path"],
        ref = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "keys_only" / "cmp" / wc.rel_path / "part.parquet")]["ref_paths"],
    output:
        str(OUTPUT_ROOT / "keys_only" / "cmp") + "/{rel_path}/part.parquet",
    log:
        str(OUTPUT_ROOT / "log" / "keys_only" / "cmp") + "/{rel_path}/run.log",
    params:
        job = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "keys_only" / "cmp" / wc.rel_path / "part.parquet")],
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        _compute_keys_only(*_preprocess(params.job)).write_parquet(output[0])


rule keys_only_ref:
    """
    All keys from ref partitions that have no cmp match, marked source="ref".
    Output mirrors the ref partition tree under keys_only/ref/.
    """
    input:
        ref = lambda wc: JOBS_REF[str(OUTPUT_ROOT / "keys_only" / "ref" / wc.rel_path / "part.parquet")]["ref_path"],
    output:
        str(OUTPUT_ROOT / "keys_only" / "ref") + "/{rel_path}/part.parquet",
    log:
        str(OUTPUT_ROOT / "log" / "keys_only" / "ref") + "/{rel_path}/run.log",
    params:
        job = lambda wc: JOBS_REF[str(OUTPUT_ROOT / "keys_only" / "ref" / wc.rel_path / "part.parquet")],
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        _compute_keys_only(*_preprocess(params.job)).write_parquet(output[0])


rule correlations:
    """Per-indicator Pearson correlation on common keys."""
    input:
        cmp = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "correlations" / "cmp" / wc.rel_path / "part.parquet")]["cmp_path"],
        ref = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "correlations" / "cmp" / wc.rel_path / "part.parquet")]["ref_paths"],
    output:
        str(OUTPUT_ROOT / "correlations" / "cmp") + "/{rel_path}/part.parquet",
    log:
        str(OUTPUT_ROOT / "log" / "correlations" / "cmp") + "/{rel_path}/run.log",
    params:
        job = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "correlations" / "cmp" / wc.rel_path / "part.parquet")],
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        _compute_correlations(*_preprocess(params.job)).write_parquet(output[0])


rule differences:
    """Absolute and relative value differences on common keys."""
    input:
        cmp = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "differences" / "cmp" / wc.rel_path / "part.parquet")]["cmp_path"],
        ref = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "differences" / "cmp" / wc.rel_path / "part.parquet")]["ref_paths"],
    output:
        str(OUTPUT_ROOT / "differences" / "cmp") + "/{rel_path}/part.parquet",
    log:
        str(OUTPUT_ROOT / "log" / "differences" / "cmp") + "/{rel_path}/run.log",
    params:
        job = lambda wc: JOBS_CMP[str(OUTPUT_ROOT / "differences" / "cmp" / wc.rel_path / "part.parquet")],
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        _compute_differences(*_preprocess(params.job)).write_parquet(output[0])

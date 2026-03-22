"""
test_hive_partition_mapper.py
-----------------------------
Unit tests for hive_partition_mapper.
"""

import time
from datetime import date, datetime, timedelta

import pytest

from hive_partition_mapper import (
    Dataset,
    Granularity,
    make_partitioned_file,
    map_datasets,
    iter_pairs,
    parse_time_partition,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paths_year_month():
    return [
        "s3://bucket-a/data/year=2024/month=1/part.parquet",
        "s3://bucket-a/data/year=2024/month=2/part.parquet",
        "s3://bucket-a/data/year=2024/month=3/part.parquet",
    ]

@pytest.fixture
def paths_year_month_day():
    return [
        "s3://bucket-b/data/year=2024/month=1/day=15/part.parquet",
        "s3://bucket-b/data/year=2024/month=1/day=31/part.parquet",
        "s3://bucket-b/data/year=2024/month=2/day=1/part.parquet",
        "s3://bucket-b/data/year=2024/month=2/day=28/part.parquet",
        "s3://bucket-b/data/year=2024/month=3/day=5/part.parquet",
    ]

@pytest.fixture
def paths_year_date():
    return [
        "s3://bucket-b/data/year=2024/date=20240115/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240131/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240201/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240228/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240305/part.parquet",
    ]

@pytest.fixture
def paths_year_quarter():
    return [
        "s3://bucket-b/data/year=2024/quarter=1/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=2/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=3/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=4/part.parquet",
    ]

@pytest.fixture
def myotherapp_paths():
    return [
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=1",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=2",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=3",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=4",
    ]

@pytest.fixture
def myapp_paths():
    return [
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=01/day=01",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=02/day=01",
        "/mnt/data/dataset/V3//5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=07",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=08",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=11/day=09",
    ]

@pytest.fixture
def myapp_paths_2():
    return [
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=PROUT/year=2024/month=01/day=01",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=PROUT/year=2024/month=02/day=01",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=PROUT/year=2024/month=04/day=07",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=PROUT/year=2024/month=04/day=08",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=PROUT/year=2024/month=11/day=09",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=01/day=01",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=02/day=01",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=07",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=08",
        "/mnt/data/dataset/V4/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=11/day=09",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pairs(a_paths, b_paths):
    return list(iter_pairs(map_datasets(
        Dataset.from_paths("A", a_paths),
        Dataset.from_paths("B", b_paths),
    )))

def pair_dates(a_paths, b_paths):
    return [((s.start, s.end), (t.start, t.end)) for s, t in pairs(a_paths, b_paths)]


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------

class TestParsing:

    def test_year_month(self):
        start, end, gran = parse_time_partition("year=2024/month=3/part.parquet")
        assert start == datetime(2024, 3, 1)
        assert end   == datetime(2024, 4, 1)
        assert gran  == Granularity.MONTH

    def test_year_month_day(self):
        start, end, gran = parse_time_partition("year=2024/month=1/day=15/part.parquet")
        assert start == datetime(2024, 1, 15)
        assert end   == datetime(2024, 1, 16)
        assert gran  == Granularity.DAY

    def test_year_date(self):
        start, end, gran = parse_time_partition("year=2024/date=20240115/part.parquet")
        assert start == datetime(2024, 1, 15)
        assert end   == datetime(2024, 1, 16)
        assert gran  == Granularity.DAY

    def test_year_month_day_and_year_date_equivalent(self):
        r1 = parse_time_partition("year=2024/month=01/day=15/part.parquet")
        r2 = parse_time_partition("year=2024/date=20240115/part.parquet")
        assert r1[:2] == r2[:2]

    @pytest.mark.parametrize("q,exp_start,exp_end", [
        (1, datetime(2024, 1,  1), datetime(2024, 4,  1)),
        (2, datetime(2024, 4,  1), datetime(2024, 7,  1)),
        (3, datetime(2024, 7,  1), datetime(2024, 10, 1)),
        (4, datetime(2024, 10, 1), datetime(2025, 1,  1)),
    ])
    def test_year_quarter(self, q, exp_start, exp_end):
        start, end, gran = parse_time_partition(f"year=2024/quarter={q}/part.parquet")
        assert start == exp_start
        assert end   == exp_end
        assert gran  == Granularity.QUARTER

    def test_year_only(self):
        start, end, gran = parse_time_partition("year=2024/part.parquet")
        assert start == datetime(2024, 1, 1)
        assert end   == datetime(2025, 1, 1)
        assert gran  == Granularity.YEAR

    def test_december_month_wraps_year(self):
        _, end, _ = parse_time_partition("year=2024/month=12/part.parquet")
        assert end == datetime(2025, 1, 1)

    def test_no_time_partition_raises(self):
        with pytest.raises(ValueError):
            parse_time_partition("s3://bucket/data/instrument=X/part.parquet")

    def test_extra_keys_extracted(self):
        f = make_partitioned_file(
            "/data/instrument_group=futures/indicator_group=TOTO/year=2024/month=01/day=01"
        )
        assert f.extra_keys == {"instrument_group": "futures", "indicator_group": "TOTO"}

    @pytest.mark.parametrize("key", ["year", "month", "day", "quarter", "date"])
    def test_time_keys_not_in_extra_keys(self, key):
        f = make_partitioned_file("year=2024/month=1/day=5/part.parquet")
        assert key not in f.extra_keys

    def test_double_slash_in_path_ignored(self):
        f = make_partitioned_file(
            "/mnt/data/dataset/V3//5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=07"
        )
        assert f.start == datetime(2024, 4, 7)
        assert f.granularity == Granularity.DAY


# ---------------------------------------------------------------------------
# 2. Dataset deduplication
# ---------------------------------------------------------------------------

class TestDatasetDedup:

    def test_dedup_same_format(self):
        path = "year=2024/month=1/day=15/part.parquet"
        ds = Dataset.from_paths("A", [path, path])
        assert len(ds.files) == 1

    def test_dedup_equivalent_formats(self):
        ds = Dataset.from_paths("A", [
            "year=2024/month=01/day=15/part.parquet",
            "year=2024/date=20240115/part.parquet",
        ])
        assert len(ds.files) == 1

    def test_no_dedup_different_extra_keys(self):
        ds = Dataset.from_paths("A", [
            "indicator_group=TOTO/year=2024/month=1/day=1/part.parquet",
            "indicator_group=PROUT/year=2024/month=1/day=1/part.parquet",
        ])
        assert len(ds.files) == 2

    def test_files_sorted_by_start(self, paths_year_month):
        ds = Dataset.from_paths("A", paths_year_month[::-1])
        starts = [f.start for f in ds.files]
        assert starts == sorted(starts)

    def test_myapp_mixed_groups_all_kept(self, myapp_paths_2):
        assert len(Dataset.from_paths("B", myapp_paths_2).files) == 10


# ---------------------------------------------------------------------------
# 3. Basic granularity matching
# ---------------------------------------------------------------------------

class TestBasicMatching:

    def test_month_vs_day_pair_count(self, paths_year_month, paths_year_month_day):
        assert len(pairs(paths_year_month, paths_year_month_day)) == 5

    def test_month_vs_date_pair_count(self, paths_year_month, paths_year_date):
        assert len(pairs(paths_year_month, paths_year_date)) == 5

    def test_month_vs_day_and_date_equivalent(self, paths_year_month, paths_year_month_day, paths_year_date):
        assert sorted(pair_dates(paths_year_month, paths_year_month_day)) == \
               sorted(pair_dates(paths_year_month, paths_year_date))

    def test_month_vs_quarter_pair_count(self, paths_year_month, paths_year_quarter):
        result = pairs(paths_year_month, paths_year_quarter)
        assert len(result) == 3
        assert all(tgt.start == datetime(2024, 1, 1) for _, tgt in result)

    def test_quarter_vs_month_symmetric(self, paths_year_month, paths_year_quarter):
        p1 = pair_dates(paths_year_month,   paths_year_quarter)
        p2 = pair_dates(paths_year_quarter, paths_year_month)
        assert sorted((b, a) for a, b in p1) == sorted(p2)

    def test_day_vs_month_many_to_one(self, paths_year_month_day, paths_year_month):
        result = pairs(paths_year_month_day, paths_year_month)
        assert len(result) == 5
        assert all(tgt.granularity == Granularity.MONTH for _, tgt in result)

    def test_no_match_non_overlapping(self):
        assert pairs(["s3://a/year=2023/month=6/part.parquet"],
                     ["s3://b/year=2025/month=6/part.parquet"]) == []

    def test_exact_boundary_no_overlap(self):
        assert pairs(["s3://a/year=2024/month=1/part.parquet"],
                     ["s3://b/year=2024/month=2/part.parquet"]) == []

    def test_empty_dataset_a(self, paths_year_month):
        assert pairs([], paths_year_month) == []

    def test_empty_dataset_b(self, paths_year_month):
        assert pairs(paths_year_month, []) == []


# ---------------------------------------------------------------------------
# 4. Extra-key filtering
# ---------------------------------------------------------------------------

class TestExtraKeyFiltering:

    def test_toto_vs_prout_no_match(self, myapp_paths_2):
        toto  = [p for p in myapp_paths_2 if "TOTO"  in p]
        prout = [p for p in myapp_paths_2 if "PROUT" in p]
        assert pairs(toto, prout) == []

    def test_toto_vs_toto_matches(self, myapp_paths, myapp_paths_2):
        toto_v4 = [p for p in myapp_paths_2 if "TOTO" in p]
        assert len(pairs(myapp_paths, toto_v4)) == 5

    def test_myapp_v3_vs_mixed_v4_only_toto_matches(self, myapp_paths, myapp_paths_2):
        result = pairs(myapp_paths, myapp_paths_2)
        assert len(result) == 5
        assert all(s.extra_keys["indicator_group"] == "TOTO" for s, _ in result)
        assert all(t.extra_keys["indicator_group"] == "TOTO" for _, t in result)

    def test_myotherapp_no_extra_keys_matches_all_groups(self, myotherapp_paths, myapp_paths_2):
        assert len(pairs(myotherapp_paths, myapp_paths_2)) == 10

    def test_myotherapp_vs_myapp_toto_only(self, myotherapp_paths, myapp_paths):
        assert len(pairs(myotherapp_paths, myapp_paths)) == 5

    @pytest.mark.parametrize("myapp_day,expected_quarter_start", [
        (datetime(2024,  1,  1), datetime(2024,  1, 1)),  # Q1
        (datetime(2024,  2,  1), datetime(2024,  1, 1)),  # Q1
        (datetime(2024,  4,  7), datetime(2024,  4, 1)),  # Q2
        (datetime(2024,  4,  8), datetime(2024,  4, 1)),  # Q2
        (datetime(2024, 11,  9), datetime(2024, 10, 1)),  # Q4
    ])
    def test_myotherapp_vs_myapp_correct_quarters(self, myotherapp_paths, myapp_paths, myapp_day, expected_quarter_start):
        matched = {t.start: s.start for s, t in pairs(myotherapp_paths, myapp_paths)}
        assert matched[myapp_day] == expected_quarter_start

    def test_shared_key_different_value_no_match(self):
        assert pairs(
            ["/data/instrument_group=futures/year=2024/month=1/day=1"],
            ["/data/instrument_group=equities/year=2024/month=1/day=1"],
        ) == []

    def test_shared_key_same_value_matches(self):
        assert len(pairs(
            ["/data/instrument_group=futures/year=2024/month=1/day=1"],
            ["/data/instrument_group=futures/year=2024/month=1/day=1/part.parquet"],
        )) == 1

    def test_disjoint_extra_keys_match_on_time(self):
        assert len(pairs(
            ["/data/region=EU/year=2024/month=1/part.parquet"],
            ["/data/indicator_group=TOTO/year=2024/month=1/part.parquet"],
        )) == 1


# ---------------------------------------------------------------------------
# 5. Interval correctness
# ---------------------------------------------------------------------------

class TestIntervalCorrectness:

    def test_month_vs_day_correct_intervals(self, paths_year_month, paths_year_month_day):
        for src, tgt in pairs(paths_year_month, paths_year_month_day):
            assert tgt.start >= src.start
            assert tgt.end   <= src.end

    def test_quarter_contains_months(self, paths_year_quarter, paths_year_month):
        for src, tgt in pairs(paths_year_quarter, paths_year_month):
            assert tgt.start >= src.start
            assert tgt.end   <= src.end

    def test_no_duplicate_pairs(self, paths_year_month, paths_year_month_day):
        result = pairs(paths_year_month, paths_year_month_day)
        seen = [(s.path, t.path) for s, t in result]
        assert len(seen) == len(set(seen))

    def test_no_duplicate_pairs_mixed_formats(self, paths_year_month, paths_year_month_day, paths_year_date):
        assert len(pairs(paths_year_month, paths_year_month_day + paths_year_date)) == 5

    def test_overlap_ratio_partial(self):
        mappings = map_datasets(
            Dataset.from_paths("A", ["s3://a/year=2024/month=1/p.parquet"]),
            Dataset.from_paths("B", [
                "s3://b/year=2024/month=1/day=1/p.parquet",
                "s3://b/year=2024/month=1/day=8/p.parquet",
                "s3://b/year=2024/month=1/day=15/p.parquet",
                "s3://b/year=2024/month=1/day=22/p.parquet",
            ]),
        )
        assert mappings[0].overlap_ratio < 1.0

    def test_overlap_ratio_full_quarter(self):
        mappings = map_datasets(
            Dataset.from_paths("A", ["s3://a/year=2024/quarter=1/p.parquet"]),
            Dataset.from_paths("B", [
                "s3://b/year=2024/month=1/p.parquet",
                "s3://b/year=2024/month=2/p.parquet",
                "s3://b/year=2024/month=3/p.parquet",
            ]),
        )
        assert mappings[0].overlap_ratio == pytest.approx(1.0)

    def test_require_full_coverage_filters(self, paths_year_month, paths_year_month_day):
        mappings = map_datasets(
            Dataset.from_paths("A", paths_year_month),
            Dataset.from_paths("B", paths_year_month_day),
            require_full_coverage=True,
        )
        assert len(mappings) == 0

    def test_require_full_coverage_passes_for_complete_quarter(self):
        mappings = map_datasets(
            Dataset.from_paths("A", ["s3://a/year=2024/quarter=1/p.parquet"]),
            Dataset.from_paths("B", [
                "s3://b/year=2024/month=1/p.parquet",
                "s3://b/year=2024/month=2/p.parquet",
                "s3://b/year=2024/month=3/p.parquet",
            ]),
            require_full_coverage=True,
        )
        assert len(mappings) == 1


# ---------------------------------------------------------------------------
# 6. Complex / combined scenarios
# ---------------------------------------------------------------------------

class TestComplexScenarios:

    def test_myotherapp_quarter_vs_myapp_day_correct_count(self, myotherapp_paths, myapp_paths):
        assert len(pairs(myotherapp_paths, myapp_paths)) == 5  # Q1(2) + Q2(2) + Q3(0) + Q4(1)

    def test_myotherapp_vs_myapp_paths_2_toto_and_prout(self, myotherapp_paths, myapp_paths_2):
        result = pairs(myotherapp_paths, myapp_paths_2)
        assert len(result) == 10
        assert {t.extra_keys["indicator_group"] for _, t in result} == {"TOTO", "PROUT"}

    def test_myapp_v3_vs_myapp_v4_same_indicator_group(self, myapp_paths, myapp_paths_2):
        result = pairs(myapp_paths, myapp_paths_2)
        assert len(result) == 5
        assert all(s.extra_keys["indicator_group"] == "TOTO" for s, _ in result)
        assert all(t.extra_keys["indicator_group"] == "TOTO" for _, t in result)

    def test_month_vs_quarter_and_day_combined(self, paths_year_month, paths_year_quarter, paths_year_month_day):
        assert len(pairs(paths_year_month, paths_year_quarter + paths_year_month_day)) == 8

    @pytest.mark.parametrize("name_a,name_b", [
        (a, b)
        for a in ["month", "day", "date", "quarter"]
        for b in ["month", "day", "date", "quarter"]
    ])
    def test_all_granularities_vs_each_other(
        self, name_a, name_b,
        paths_year_month, paths_year_month_day, paths_year_date, paths_year_quarter,
    ):
        datasets = {
            "month":   paths_year_month,
            "day":     paths_year_month_day,
            "date":    paths_year_date,
            "quarter": paths_year_quarter,
        }
        assert isinstance(pairs(datasets[name_a], datasets[name_b]), list)

    def test_myapp_q3_no_match_myotherapp_q3(self, myotherapp_paths, myapp_paths):
        mappings = map_datasets(
            Dataset.from_paths("myotherapp", myotherapp_paths),
            Dataset.from_paths("myapp",  myapp_paths),
        )
        q3 = next(m for m in mappings if m.source.start == datetime(2024, 7, 1))
        assert q3.targets == []
        assert q3.overlap_ratio == 0.0

    def test_symmetry_pair_intervals(self, paths_year_month, paths_year_month_day):
        fwd = pair_dates(paths_year_month,     paths_year_month_day)
        rev = pair_dates(paths_year_month_day, paths_year_month)
        assert sorted((b, a) for a, b in fwd) == sorted(rev)

    def test_double_slash_path_matches_normally(self, myapp_paths, myapp_paths_2):
        double_slash = [p for p in myapp_paths if "//" in p]
        normal_v4    = [p for p in myapp_paths_2 if "TOTO" in p and "04/day=07" in p]
        result = pairs(double_slash, normal_v4)
        assert len(result) == 1
        src, tgt = result[0]
        assert src.start == tgt.start == datetime(2024, 4, 7)


# ---------------------------------------------------------------------------
# 7. Large-scale / performance
# ---------------------------------------------------------------------------

INDICATOR_GROUPS  = ["TOTO", "PROUT", "ALPHA", "BETA"]
INSTRUMENT_GROUPS = ["futures", "equities", "options"]
ALL_COMBOS = [
    {"instrument_group": ig, "indicator_group": ind}
    for ig in INSTRUMENT_GROUPS
    for ind in INDICATOR_GROUPS
]  # 12 combinations

DATE_START = date(2010, 1, 1)
DATE_END   = date(2024, 12, 31)


def _daily(start, end, prefix, extra_kv=None):
    extra = "".join(f"{k}={v}/" for k, v in (extra_kv or {}).items())
    paths, d = [], start
    while d <= end:
        paths.append(f"{prefix}/{extra}year={d.year}/month={d.month:02d}/day={d.day:02d}/part.parquet")
        d += timedelta(days=1)
    return paths

def _quarterly(start_year, end_year, prefix, extra_kv=None):
    extra = "".join(f"{k}={v}/" for k, v in (extra_kv or {}).items())
    return [
        f"{prefix}/{extra}year={y}/quarter={q}/part.parquet"
        for y in range(start_year, end_year + 1) for q in range(1, 5)
    ]

def _monthly(start_year, end_year, prefix, extra_kv=None):
    extra = "".join(f"{k}={v}/" for k, v in (extra_kv or {}).items())
    return [
        f"{prefix}/{extra}year={y}/month={m:02d}/part.parquet"
        for y in range(start_year, end_year + 1) for m in range(1, 13)
    ]


class TestLargeScale:

    def test_daily_vs_quarterly(self):
        """~5.5k daily vs 60 quarterly — every day matches exactly one quarter."""
        daily     = _daily(DATE_START, DATE_END, "s3://a")
        quarterly = _quarterly(2010, 2024, "s3://b")
        t0 = time.perf_counter()
        result = pairs(daily, quarterly)
        elapsed = time.perf_counter() - t0
        print(f"\n  daily({len(daily)}) vs quarterly({len(quarterly)}): {len(result)} pairs in {elapsed:.3f}s")
        assert len(result) == len(daily)
        assert all(t.granularity == Granularity.QUARTER for _, t in result)
        assert all(s.start >= t.start and s.end <= t.end for s, t in result)
        assert elapsed < 5.0

    def test_monthly_vs_quarterly(self):
        """180 monthly vs 60 quarterly — every month maps to exactly one quarter."""
        monthly   = _monthly(2010, 2024, "s3://a")
        quarterly = _quarterly(2010, 2024, "s3://b")
        result = pairs(monthly, quarterly)
        assert len(result) == len(monthly)
        assert all(t.granularity == Granularity.QUARTER for _, t in result)

    def test_daily_vs_daily_one_to_one(self):
        """~5.5k daily A vs ~5.5k daily B — exact 1-to-1 on same dates."""
        daily_a = _daily(DATE_START, DATE_END, "s3://a")
        daily_b = _daily(DATE_START, DATE_END, "s3://b")
        t0 = time.perf_counter()
        result = pairs(daily_a, daily_b)
        elapsed = time.perf_counter() - t0
        print(f"\n  daily({len(daily_a)}) vs daily({len(daily_b)}): {len(result)} pairs in {elapsed:.3f}s")
        assert len(result) == len(daily_a)
        assert all(s.start == t.start for s, t in result)
        assert elapsed < 5.0

    def test_multigroup_daily_vs_quarterly(self):
        """~65k daily (12 groups) vs 720 quarterly (12 groups) — groups never cross."""
        daily_paths     = [p for kv in ALL_COMBOS for p in _daily(DATE_START, DATE_END, "s3://a", kv)]
        quarterly_paths = [p for kv in ALL_COMBOS for p in _quarterly(2010, 2024, "s3://b", kv)]
        t0 = time.perf_counter()
        result = pairs(daily_paths, quarterly_paths)
        elapsed = time.perf_counter() - t0
        print(f"\n  multigroup daily({len(daily_paths)}) vs quarterly({len(quarterly_paths)}): {len(result)} pairs in {elapsed:.3f}s")
        assert len(result) == len(daily_paths)
        assert all(s.extra_keys == t.extra_keys for s, t in result)
        assert elapsed < 10.0

    def test_multigroup_cross_filtering(self):
        """~5.5k TOTO daily vs ~65k all-group daily — only TOTO/futures pairs emitted."""
        toto_paths = _daily(DATE_START, DATE_END, "s3://a", {"instrument_group": "futures", "indicator_group": "TOTO"})
        all_paths  = [p for kv in ALL_COMBOS for p in _daily(DATE_START, DATE_END, "s3://b", kv)]
        t0 = time.perf_counter()
        result = pairs(toto_paths, all_paths)
        elapsed = time.perf_counter() - t0
        print(f"\n  TOTO({len(toto_paths)}) vs all-groups({len(all_paths)}): {len(result)} pairs in {elapsed:.3f}s")
        assert len(result) == len(toto_paths)
        assert all(t.extra_keys["indicator_group"]  == "TOTO"    for _, t in result)
        assert all(t.extra_keys["instrument_group"] == "futures" for _, t in result)
        assert elapsed < 10.0

    def test_no_extra_keys_vs_multigroup(self):
        """60 quarterly (no keys) vs ~65k daily (12 groups) — every day gets one quarter."""
        quarterly = _quarterly(2010, 2024, "s3://myotherapp")
        all_daily = [p for kv in ALL_COMBOS for p in _daily(DATE_START, DATE_END, "s3://myapp", kv)]
        t0 = time.perf_counter()
        result = pairs(quarterly, all_daily)
        elapsed = time.perf_counter() - t0
        print(f"\n  quarterly({len(quarterly)}) vs myapp-all({len(all_daily)}): {len(result)} pairs in {elapsed:.3f}s")
        assert len(result) == len(all_daily)
        assert elapsed < 15.0

    def test_partial_date_range_overlap(self):
        """A: 2010-2019, B: 2015-2024 — only the overlapping 2015-2019 days match."""
        daily_a = _daily(date(2010, 1, 1), date(2019, 12, 31), "s3://a")
        daily_b = _daily(date(2015, 1, 1), date(2024, 12, 31), "s3://b")
        result = pairs(daily_a, daily_b)
        overlap_days = (date(2019, 12, 31) - date(2015, 1, 1)).days + 1
        assert len(result) == overlap_days
        assert all(datetime(2015, 1, 1) <= s.start < datetime(2020, 1, 1) for s, _ in result)


# ---------------------------------------------------------------------------
# 8. from_directory
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path


def _make_tree(root: Path, structure: dict) -> None:
    """
    Recursively create a directory tree from a nested dict.
    Keys are directory/file names; a None value means create a file,
    a dict value means create a subdirectory and recurse.
    """
    for name, children in structure.items():
        path = root / name
        if children is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
            _make_tree(path, children)


@pytest.fixture
def myotherapp_tree(tmp_path):
    """
    /aggregated/
      year=2023/
        quarter=1/part.parquet  quarter=2/part.parquet
        quarter=3/part.parquet  quarter=4/part.parquet
      year=2024/
        quarter=1/part.parquet  quarter=2/part.parquet
        quarter=3/part.parquet  quarter=4/part.parquet
    """
    root = tmp_path / "aggregated"
    for year in (2023, 2024):
        for q in range(1, 5):
            _make_tree(root, {f"year={year}": {f"quarter={q}": {"part.parquet": None}}})
    return root


@pytest.fixture
def myapp_tree(tmp_path):
    """
    /5MIN_MYAPP/
      instrument_group=futures/
        indicator_group=TOTO/
          year=2024/month=01/day=01/part.parquet
          year=2024/month=02/day=01/part.parquet
          year=2024/month=04/day=07/part.parquet
        indicator_group=PROUT/
          year=2024/month=01/day=01/part.parquet
          year=2024/month=04/day=07/part.parquet
    """
    root = tmp_path / "5MIN_MYAPP"
    days = {
        "TOTO":  ["year=2024/month=01/day=01", "year=2024/month=02/day=01", "year=2024/month=04/day=07"],
        "PROUT": ["year=2024/month=01/day=01", "year=2024/month=04/day=07"],
    }
    for ind, day_paths in days.items():
        for dp in day_paths:
            _make_tree(root, {
                "instrument_group=futures": {
                    f"indicator_group={ind}": {dp: {"part.parquet": None}}
                }
            })
    return root


@pytest.fixture
def mixed_granularity_tree(tmp_path):
    """
    /data/
      year=2024/
        part.parquet            ← year-level file
        quarter=1/part.parquet  ← quarter-level files
        quarter=2/part.parquet
        month=01/part.parquet   ← month-level files
        month=01/day=15/part.parquet   ← day-level files
        month=01/day=31/part.parquet
    """
    root = tmp_path / "data"
    _make_tree(root, {
        "year=2024": {
            "part.parquet": None,
            "quarter=1": {"part.parquet": None},
            "quarter=2": {"part.parquet": None},
            "month=01":  {
                "part.parquet": None,
                "day=15": {"part.parquet": None},
                "day=31": {"part.parquet": None},
            },
        }
    })
    return root


class TestFromDirectory:

    def test_default_scans_parquet_files(self, myotherapp_tree):
        """Default behaviour: glob for parquet files, return quarter-level paths."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree)
        assert len(ds.files) == 8  # 2 years × 4 quarters
        assert all(f.granularity == Granularity.QUARTER for f in ds.files)

    def test_granularity_year(self, myotherapp_tree):
        """YEAR granularity returns one directory per year."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.YEAR)
        assert len(ds.files) == 2  # year=2023, year=2024
        assert all(f.granularity == Granularity.YEAR for f in ds.files)

    def test_granularity_year_intervals(self, myotherapp_tree):
        """YEAR directories resolve to correct [Jan 1, Jan 1+1y) intervals."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.YEAR)
        starts = sorted(f.start for f in ds.files)
        assert starts == [datetime(2023, 1, 1), datetime(2024, 1, 1)]
        for f in ds.files:
            assert f.end == datetime(f.start.year + 1, 1, 1)

    def test_granularity_quarter(self, myotherapp_tree):
        """QUARTER granularity returns quarter directories, not parquet files."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.QUARTER)
        assert len(ds.files) == 8
        assert all(f.granularity == Granularity.QUARTER for f in ds.files)
        # Paths must end with quarter=N, not part.parquet
        assert all(f.path.split("/")[-1].startswith("quarter=") for f in ds.files)

    def test_granularity_quarter_does_not_include_parquet(self, myotherapp_tree):
        """QUARTER mode must not include the parquet file paths."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.QUARTER)
        assert not any(f.path.endswith(".parquet") for f in ds.files)

    def test_granularity_day(self, myapp_tree):
        """DAY granularity returns day directories across all extra-key groups."""
        ds = Dataset.from_directory("myapp", myapp_tree, time_granularity=Granularity.DAY)
        assert len(ds.files) == 5  # 3 TOTO + 2 PROUT
        assert all(f.granularity == Granularity.DAY for f in ds.files)

    def test_granularity_day_extra_keys_preserved(self, myapp_tree):
        """Extra-key partitions (instrument_group, indicator_group) survive into extra_keys."""
        ds = Dataset.from_directory("myapp", myapp_tree, time_granularity=Granularity.DAY)
        for f in ds.files:
            assert "instrument_group" in f.extra_keys
            assert "indicator_group"  in f.extra_keys
            assert f.extra_keys["instrument_group"] == "futures"

    def test_granularity_day_groups_split_correctly(self, myapp_tree):
        """TOTO and PROUT end up in separate groups."""
        ds = Dataset.from_directory("myapp", myapp_tree, time_granularity=Granularity.DAY)
        toto  = [f for f in ds.files if f.extra_keys["indicator_group"] == "TOTO"]
        prout = [f for f in ds.files if f.extra_keys["indicator_group"] == "PROUT"]
        assert len(toto)  == 3
        assert len(prout) == 2

    def test_files_sorted_by_start(self, myotherapp_tree):
        """from_directory must return files sorted by start datetime."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.QUARTER)
        starts = [f.start for f in ds.files]
        assert starts == sorted(starts)

    def test_default_none_is_unchanged(self, myotherapp_tree):
        """Explicit time_granularity=None behaves identically to the default."""
        ds1 = Dataset.from_directory("a", myotherapp_tree)
        ds2 = Dataset.from_directory("b", myotherapp_tree, time_granularity=None)
        assert [(f.start, f.end) for f in ds1.files] == [(f.start, f.end) for f in ds2.files]

    def test_mixed_granularity_tree_year_only(self, mixed_granularity_tree):
        """In a mixed-depth tree, YEAR returns only the year-level directory."""
        ds = Dataset.from_directory("data", mixed_granularity_tree, time_granularity=Granularity.YEAR)
        assert len(ds.files) == 1
        assert ds.files[0].granularity == Granularity.YEAR
        assert ds.files[0].start == datetime(2024, 1, 1)

    def test_mixed_granularity_tree_quarter_only(self, mixed_granularity_tree):
        """QUARTER mode returns only quarter directories, not year/month/day."""
        ds = Dataset.from_directory("data", mixed_granularity_tree, time_granularity=Granularity.QUARTER)
        assert len(ds.files) == 2
        assert all(f.granularity == Granularity.QUARTER for f in ds.files)

    def test_mixed_granularity_tree_month_only(self, mixed_granularity_tree):
        """MONTH mode returns only the month directory."""
        ds = Dataset.from_directory("data", mixed_granularity_tree, time_granularity=Granularity.MONTH)
        assert len(ds.files) == 1
        assert ds.files[0].granularity == Granularity.MONTH
        assert ds.files[0].start == datetime(2024, 1, 1)

    def test_mixed_granularity_tree_day_only(self, mixed_granularity_tree):
        """DAY mode returns only the day directories."""
        ds = Dataset.from_directory("data", mixed_granularity_tree, time_granularity=Granularity.DAY)
        assert len(ds.files) == 2
        assert all(f.granularity == Granularity.DAY for f in ds.files)
        starts = sorted(f.start for f in ds.files)
        assert starts == [datetime(2024, 1, 15), datetime(2024, 1, 31)]

    def test_empty_directory(self, tmp_path):
        """Empty root returns an empty dataset without error."""
        ds = Dataset.from_directory("empty", tmp_path, time_granularity=Granularity.QUARTER)
        assert ds.files == []

    def test_no_matching_granularity(self, myotherapp_tree):
        """Requesting DAY granularity from a quarter-only tree returns empty."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, time_granularity=Granularity.DAY)
        assert ds.files == []

    def test_custom_glob_still_works(self, myotherapp_tree):
        """The default glob=... parameter still works when time_granularity is None."""
        ds = Dataset.from_directory("myotherapp", myotherapp_tree, glob="**/quarter=*/part.parquet")
        assert len(ds.files) == 8
        assert all(f.granularity == Granularity.QUARTER for f in ds.files)

    def test_from_directory_matches_from_paths(self, myapp_tree):
        """from_directory(DAY) and from_paths with equivalent paths produce same intervals."""
        ds_dir = Dataset.from_directory("myapp", myapp_tree, time_granularity=Granularity.DAY)
        manual_paths = [
            str(p) for p in myapp_tree.rglob("day=*") if p.is_dir()
        ]
        ds_paths = Dataset.from_paths("myapp", manual_paths)
        dir_intervals   = sorted((f.start, f.end, tuple(sorted(f.extra_keys.items()))) for f in ds_dir.files)
        paths_intervals = sorted((f.start, f.end, tuple(sorted(f.extra_keys.items()))) for f in ds_paths.files)
        assert dir_intervals == paths_intervals

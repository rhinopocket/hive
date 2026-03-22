"""
hive_partition_mapper.py
------------------------
Maps files between two hive-partitioned datasets whose time partitions may have
different granularities (year, quarter, month, day).

Supported partition formats (at the end of any path):
  year=YYYY
  year=YYYY/month=M[M]
  year=YYYY/quarter=Q
  year=YYYY/month=M[M]/day=D[D]
  date=YYYYMMDD
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# 1. Granularity
# ---------------------------------------------------------------------------

class Granularity(Enum):
    YEAR    = auto()
    QUARTER = auto()
    MONTH   = auto()
    DAY     = auto()


# ---------------------------------------------------------------------------
# 2. Parsing: path → (non-time prefix, start datetime, granularity)
# ---------------------------------------------------------------------------

# date= pattern tried before key=value patterns
_DATE_STR_RE = re.compile(r"(?:^|/)date=(\d{4})(\d{2})(\d{2})(?:/|$)")

_KV_RE = re.compile(r"(?:^|/)(\w+)=([^/]+)")


def _kv(path: str) -> dict[str, str]:
    """Extract all key=value pairs from a hive path."""
    return {m.group(1).lower(): m.group(2) for m in _KV_RE.finditer(path)}


def parse_time_partition(path: str) -> tuple[datetime, datetime, Granularity]:
    """
    Parse the time partition at the end of a hive path.

    Returns
    -------
    (start, end, granularity)
        start is inclusive, end is exclusive.
    """
    # --- date=YYYYMMDD style ---
    m = _DATE_STR_RE.search(path)
    if m:
        yr, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        start = datetime(yr, mo, dy)
        return start, start + timedelta(days=1), Granularity.DAY

    kv = _kv(path)

    year = int(kv["year"]) if "year" in kv else None
    if year is None:
        raise ValueError(f"No recognisable time partition found in: {path!r}")

    # year / month / day
    if "day" in kv:
        month = int(kv["month"])
        day   = int(kv["day"])
        start = datetime(year, month, day)
        return start, start + timedelta(days=1), Granularity.DAY

    # year / month
    if "month" in kv:
        month = int(kv["month"])
        start = datetime(year, month, 1)
        # next month
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        return start, end, Granularity.MONTH

    # year / quarter
    if "quarter" in kv:
        q     = int(kv["quarter"])
        month = (q - 1) * 3 + 1          # Q1→1, Q2→4, Q3→7, Q4→10
        start = datetime(year, month, 1)
        end_m = month + 3
        end   = datetime(year + 1, 1, 1) if end_m > 12 else datetime(year, end_m, 1)
        return start, end, Granularity.QUARTER

    # year only
    return datetime(year, 1, 1), datetime(year + 1, 1, 1), Granularity.YEAR


# ---------------------------------------------------------------------------
# 3. PartitionedFile — a file with its resolved time interval
# ---------------------------------------------------------------------------

@dataclass
class PartitionedFile:
    path: str
    start: datetime
    end: datetime          # exclusive
    granularity: Granularity
    # non-time key=value pairs (dataset_id, region, …)
    extra_keys: dict[str, str] = field(default_factory=dict)

    # ---- convenience ----

    def overlaps(self, other: "PartitionedFile") -> bool:
        return self.start < other.end and other.start < self.end

    def __repr__(self) -> str:
        return (
            f"PartitionedFile({self.path!r}, "
            f"{self.start:%Y-%m-%d %H:%M} → {self.end:%Y-%m-%d %H:%M}, "
            f"{self.granularity.name})"
        )


_TIME_KEYS = {"year", "month", "day", "quarter", "date"}


def make_partitioned_file(path: str) -> PartitionedFile:
    """Build a PartitionedFile from a raw path string."""
    start, end, gran = parse_time_partition(path)
    extra = {k: v for k, v in _kv(path).items() if k not in _TIME_KEYS}
    return PartitionedFile(path=path, start=start, end=end,
                           granularity=gran, extra_keys=extra)


# ---------------------------------------------------------------------------
# 4. Dataset — a collection of PartitionedFiles
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    files: list[PartitionedFile] = field(default_factory=list)

    @classmethod
    def from_paths(cls, name: str, paths: list[str]) -> "Dataset":
        # Deduplicate by (start, end): keep the first path seen for each
        # interval so that date=YYYYMMDD and year=/month=/day= variants of the
        # same partition don't produce duplicate files.
        seen: dict[tuple, PartitionedFile] = {}
        for p in paths:
            f = make_partitioned_file(p)
            # Include extra_keys in the dedup key so that files at the same
            # time interval but different non-time partitions (e.g. different
            # indicator_group values) are kept as distinct entries.
            key = (f.start, f.end, tuple(sorted(f.extra_keys.items())))
            if key not in seen:
                seen[key] = f
        files = sorted(seen.values(), key=lambda f: f.start)
        return cls(name=name, files=files)

    @classmethod
    def from_directory(
        cls,
        name: str,
        root: str | Path,
        glob: str = "**/*.parquet",
        time_granularity: Granularity | None = None,
    ) -> "Dataset":
        """Scan a local directory for partitioned files.

        Parameters
        ----------
        name
            Dataset name.
        root
            Root directory to scan.
        glob
            File glob pattern used when time_granularity is None (default).
            Ignored when time_granularity is set.
        time_granularity
            If given, collect partition *directories* at exactly this
            granularity level instead of scanning for files.  Paths that
            cannot be parsed or whose granularity does not match are skipped.

            Example: root="/data/myotherapp/", time_granularity=Granularity.YEAR
            will collect "/data/myotherapp/year=2024" but not its sub-directories.
        """
        root = Path(root)

        if time_granularity is None:
            paths = [str(p) for p in root.glob(glob)]
            return cls.from_paths(name, paths)

        # Glob patterns that reach exactly the right partition depth.
        # Non-time (extra) key directories may appear before the time ones,
        # so we use ** to allow any number of leading key=value segments.
        _GRAN_GLOBS: dict[Granularity, list[str]] = {
            Granularity.YEAR:    ["**/year=*"],
            Granularity.QUARTER: ["**/year=*/quarter=*"],
            Granularity.MONTH:   ["**/year=*/month=*"],
            Granularity.DAY:     [
                "**/year=*/month=*/day=*",
                "**/year=*/date=*",
            ],
        }

        paths: list[str] = []
        for pattern in _GRAN_GLOBS[time_granularity]:
            for p in root.glob(pattern):
                if not p.is_dir():
                    continue
                try:
                    _, _, gran = parse_time_partition(str(p))
                except ValueError:
                    continue
                if gran == time_granularity:
                    paths.append(str(p))

        return cls.from_paths(name, paths)


# ---------------------------------------------------------------------------
# 5. Mapper — the core matching logic
# ---------------------------------------------------------------------------

@dataclass
class FileMapping:
    """One file from dataset A mapped to ≥1 overlapping files from dataset B."""
    source: PartitionedFile
    targets: list[PartitionedFile]

    @cached_property
    def overlap_ratio(self) -> float:
        """
        Fraction of *source* interval that is covered by the union of targets.
        Useful to detect gaps (ratio < 1) or identify coarse→fine mappings.
        """
        if not self.targets:
            return 0.0
        # merge target intervals that overlap with source
        merged: list[tuple[datetime, datetime]] = []
        for t in sorted(self.targets, key=lambda x: x.start):
            lo = max(t.start, self.source.start)
            hi = min(t.end,   self.source.end)
            if lo >= hi:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        covered = sum((hi - lo).total_seconds() for lo, hi in merged)
        total   = (self.source.end - self.source.start).total_seconds()
        return covered / total if total else 0.0

    def __repr__(self) -> str:
        ratio = self.overlap_ratio
        return (
            f"FileMapping(\n"
            f"  source : {self.source}\n"
            f"  targets: {len(self.targets)} file(s)  "
            f"coverage={ratio:.0%}\n"
            + "".join(f"    → {t}\n" for t in self.targets)
            + ")"
        )


def _group_by_extra_keys(
    files: list[PartitionedFile],
) -> dict[tuple, list[PartitionedFile]]:
    """Bucket files by their non-time partition fingerprint."""
    groups: dict[tuple, list[PartitionedFile]] = {}
    for f in files:
        key = tuple(sorted(f.extra_keys.items()))
        groups.setdefault(key, []).append(f)
    return groups


def _compatible_group_pairs(
    groups_a: dict[tuple, list[PartitionedFile]],
    groups_b: dict[tuple, list[PartitionedFile]],
) -> Iterator[tuple[tuple, tuple]]:
    """
    Yield every (key_a, key_b) pair whose extra-key fingerprints are
    compatible (shared keys agree on value; disjoint key sets always match).

    Uses a positive index on B instead of the naive O(G_A × G_B) scan:

      1. Build a positive index:
         b_index[(k, v)] = set of B fingerprints that have key k = v

      2. For each A fingerprint, start with the full set of B fingerprints,
         then for each key k=v_a that A owns, subtract all B fingerprints
         that have key k with any value other than v_a
         (i.e. union of b_index[(k, v)] for all v ≠ v_a).

      3. What remains is exactly the compatible B fingerprints.

    Complexity: O(G_B × K_B) to build, O(G_A × K_A × V) to query — where K
    is the number of key dimensions and V the average number of distinct values
    per dimension.  In practice K and V are small constants, giving O(G_A + G_B).
    """
    all_b_keys = set(groups_b.keys())

    # b_index[(k, v)] = B fingerprints where dimension k has value v (positive index)
    b_index: dict[tuple[str, str], set[tuple]] = {}
    for key_b in groups_b:
        for k, v in key_b:
            b_index.setdefault((k, v), set()).add(key_b)

    # b_keys_with_dim[k] = all B fingerprints that define dimension k at all
    b_keys_with_dim: dict[str, set[tuple]] = {}
    for key_b in groups_b:
        for k, _ in key_b:
            b_keys_with_dim.setdefault(k, set()).add(key_b)

    for key_a in groups_a:
        compatible = all_b_keys.copy()
        for k, v_a in key_a:
            # Remove B groups that have dimension k but with a different value.
            # = (all B groups that define k) minus (B groups where k == v_a)
            if k in b_keys_with_dim:
                incompatible = b_keys_with_dim[k] - b_index.get((k, v_a), set())
                compatible -= incompatible
        for key_b in compatible:
            yield key_a, key_b


def _merge_overlap(
    a_files: list[PartitionedFile],
    b_files: list[PartitionedFile],
) -> dict[str, list[PartitionedFile]]:
    """
    Two-pointer sweep over two sorted, non-overlapping sequences.
    Returns {source_path: [overlapping targets]}.
    O(N + M) — exploits the invariant that intervals within a single
    dataset never overlap (guaranteed by dedup in Dataset.from_paths).
    """
    result: dict[str, list[PartitionedFile]] = {f.path: [] for f in a_files}
    i = j = 0
    while i < len(a_files) and j < len(b_files):
        a, b = a_files[i], b_files[j]
        if a.end <= b.start:    # a is entirely before b — advance a
            i += 1
        elif b.end <= a.start:  # b is entirely before a — advance b
            j += 1
        else:
            # a and b overlap. Scan forward in B collecting all Bs that
            # overlap this A, then advance A. Each B is visited at most
            # twice (once as the current j, once as part of a scan), so
            # the total work across all iterations is O(N + M + P).
            k = j
            while k < len(b_files) and b_files[k].start < a.end:
                if b_files[k].end > a.start:  # genuine overlap
                    result[a.path].append(b_files[k])
                k += 1
            i += 1
            # Rewind j to the first B that could still overlap the next A.
            # Since intervals in A are non-overlapping and sorted, next A
            # starts at or after current A's start, so we only need to drop
            # Bs that ended before current A's start.
            while j < len(b_files) and b_files[j].end <= a.start:
                j += 1
    return result


def map_datasets(
    dataset_a: Dataset,
    dataset_b: Dataset,
    *,
    require_full_coverage: bool = False,
) -> list[FileMapping]:
    """
    For every file in dataset_a, find all files in dataset_b whose time
    interval overlaps.

    Complexity: O(N + M + P) for the sweep; O(G_A + G_B) for group
    matching via inverted index (down from O(G_A × G_B) naive scan).

    Parameters
    ----------
    dataset_a, dataset_b
        The two datasets to map.
    require_full_coverage
        If True, only include mappings where overlap_ratio == 1.0
        (every moment of the source is covered by at least one target).
    """
    groups_a = _group_by_extra_keys(dataset_a.files)
    groups_b = _group_by_extra_keys(dataset_b.files)

    all_targets: dict[str, list[PartitionedFile]] = {
        f.path: [] for f in dataset_a.files
    }
    for key_a, key_b in _compatible_group_pairs(groups_a, groups_b):
        matched = _merge_overlap(groups_a[key_a], groups_b[key_b])
        for path, targets in matched.items():
            all_targets[path].extend(targets)

    mappings: list[FileMapping] = []
    for src in dataset_a.files:
        mapping = FileMapping(source=src, targets=all_targets[src.path])
        if require_full_coverage and mapping.overlap_ratio < 1.0:
            continue
        mappings.append(mapping)
    return mappings


def iter_pairs(
    mappings: list[FileMapping],
) -> Iterator[tuple[PartitionedFile, PartitionedFile]]:
    """
    Flatten a list of FileMapping into (source, target) pairs.
    Pairs are deduplicated by (source interval, target interval) so that
    datasets mixing date=YYYYMMDD and year=/month=/day= formats don't produce
    duplicate pairs.
    """
    for m in mappings:
        yield from ((m.source, t) for t in m.targets)


# ---------------------------------------------------------------------------
# 6. Quick summary helper
# ---------------------------------------------------------------------------

def summarize(mappings: list[FileMapping]) -> str:
    if not mappings:
        return "No mappings found."
    total_src  = len(mappings)
    with_match = sum(1 for m in mappings if m.targets)
    total_tgt  = sum(len(m.targets) for m in mappings)
    avg        = total_tgt / total_src if total_src else 0
    lines = [
        f"Source files     : {total_src}",
        f"With ≥1 match    : {with_match}",
        f"Unmatched        : {total_src - with_match}",
        f"Total target refs: {total_tgt}",
        f"Avg targets/src  : {avg:.1f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Example / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    paths_year_month = [
        "s3://bucket-a/data/year=2024/month=1/part.parquet",
        "s3://bucket-a/data/year=2024/month=2/part.parquet",
        "s3://bucket-a/data/year=2024/month=3/part.parquet",
    ]
    paths_year_month_day = [
        "s3://bucket-b/data/year=2024/month=1/day=15/part.parquet",
        "s3://bucket-b/data/year=2024/month=1/day=31/part.parquet",
        "s3://bucket-b/data/year=2024/month=2/day=1/part.parquet",
        "s3://bucket-b/data/year=2024/month=2/day=28/part.parquet",
        "s3://bucket-b/data/year=2024/month=3/day=5/part.parquet",
    ]

    paths_year_date = [
        "s3://bucket-b/data/year=2024/date=20240115/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240131/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240201/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240228/part.parquet",
        "s3://bucket-b/data/year=2024/date=20240305/part.parquet",
    ]
    paths_year_quarter = [
        "s3://bucket-b/data/year=2024/quarter=1/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=2/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=3/part.parquet",
        "s3://bucket-b/data/year=2024/quarter=4/part.parquet",
    ]

    myotherapp_paths = [
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=1",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=2",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=3",
        "/mnt/data/dataset/5MIN_MY_OTHER_APP/aggregated/year=2024/quarter=4",
    ]

    myapp_paths = [
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=01/day=01",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=02/day=01",
        "/mnt/data/dataset/V3//5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=07",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=04/day=08",
        "/mnt/data/dataset/V3/5MIN_MYAPP_SIMU/instrument_group=futures/indicator_group=TOTO/year=2024/month=11/day=09",
    ]
    myapp_paths_2 = [
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
    ds_year_month = Dataset.from_paths("monthly", paths_year_month)
    ds_year_month_day = Dataset.from_paths("daily", paths_year_month_day)
    ds_year_date = Dataset.from_paths("date", paths_year_date)
    ds_year_quarter = Dataset.from_paths("date", paths_year_quarter)
    ds_myotherapp = Dataset.from_paths("myotherapp", myotherapp_paths)
    ds_myapp = Dataset.from_paths("myapp", myapp_paths)
    ds_myapp_2 = Dataset.from_paths("myapp2", myapp_paths_2)
    #mappings = map_datasets(ds_year_date, ds_year_quarter)
    #mappings = map_datasets(ds_year_quarter, ds_year_date)
    #mappings = map_datasets(ds_myotherapp, ds_myapp)
    mappings = map_datasets(ds_myapp_2, ds_myapp)

    print("=== Mapping summary ===")
    print(summarize(mappings))
    print()
    for m in mappings:
        print(m)
    print("\n=== Flat pairs ===")
    for src, tgt in iter_pairs(mappings):
        print(f"{src.path} --> {tgt.path}")
        #print(f"  {Path(src.path).parent.name}  ←→  {Path(tgt.path).parent.name}")

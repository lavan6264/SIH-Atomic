"""Apply normalize_description() to all 4 CPSE material CSVs.

Loads data/{cpse}_materials.csv, adds a normalized_description column, and
writes data/{cpse}_materials_normalized.csv. Then prints a before/after
sample picked directly from the data: rows whose normalized_description is
shared by more than one CPSE, proving convergence actually happened on
real fragmented rows rather than on hand-picked examples.
"""

from pathlib import Path

import pandas as pd

from pipeline.normalize import normalize_description

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CPSES = ["iocl", "ongc", "hpcl", "bpcl"]
SAMPLE_SIZE = 10


def main():
    combined = []
    for cpse in CPSES:
        in_path = DATA_DIR / f"{cpse}_materials.csv"
        df = pd.read_csv(in_path)
        df["raw_description"] = df["raw_description"].fillna("")
        df["normalized_description"] = df["raw_description"].apply(normalize_description)

        out_path = DATA_DIR / f"{cpse}_materials_normalized.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path.name} ({len(df)} rows)")

        df["cpse"] = cpse.upper()
        combined.append(df)

    all_df = pd.concat(combined, ignore_index=True)

    # Groups where normalization pulled rows from >=2 different CPSEs onto
    # the same normalized text -- these are the fragmentation cases we care
    # about demonstrating.
    distinct_cpse_counts = all_df.groupby("normalized_description")["cpse"].nunique()
    cross_cpse_groups = distinct_cpse_counts[distinct_cpse_counts >= 2].sort_values(ascending=False)

    print(f"\n{len(cross_cpse_groups)} normalized descriptions are shared across 2+ CPSEs "
          f"(out of {all_df['normalized_description'].nunique()} distinct normalized descriptions).")

    print(f"\nBefore/after normalization -- {SAMPLE_SIZE} sample rows picked for cross-CPSE convergence:\n")
    shown = 0
    for norm_desc in cross_cpse_groups.index:
        if shown >= SAMPLE_SIZE:
            break
        subset = all_df[all_df["normalized_description"] == norm_desc].drop_duplicates(subset="cpse")
        print(f'>> converged normalized form: "{norm_desc}"')
        for _, row in subset.iterrows():
            if shown >= SAMPLE_SIZE:
                break
            print(f"   [{row['cpse']:4s}] {row['raw_description']!r:45s} -> {row['normalized_description']!r}")
            shown += 1
        print()


if __name__ == "__main__":
    main()

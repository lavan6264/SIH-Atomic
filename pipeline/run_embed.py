"""Build the Chroma vector store and sanity-check embedding similarity.

1. Loads + combines the 4 normalized CPSE CSVs.
2. Builds/updates the persisted "materials" Chroma collection.
3. Re-derives the same cross-CPSE convergence groups pipeline/run_normalize.py
   found (normalized_description shared by 2+ CPSEs), picks 5 of them, and
   queries find_similar() with one CPSE's raw text to see whether the other
   CPSEs' rows for the SAME item come back as high-similarity top-k matches.
4. Runs find_similar() on 2 CPSE-unique ("true negative") items to confirm
   they don't falsely match other CPSEs' unrelated items.
"""

from pathlib import Path

import pandas as pd

from data.generate_dataset import UNIQUE_SPLIT
from pipeline.embed import build_vector_store, find_similar

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CPSES = ["iocl", "ongc", "hpcl", "bpcl"]


def load_combined() -> pd.DataFrame:
    frames = []
    for cpse in CPSES:
        path = DATA_DIR / f"{cpse}_materials_normalized.csv"
        df = pd.read_csv(path)
        df["raw_description"] = df["raw_description"].fillna("")
        df["normalized_description"] = df["normalized_description"].fillna("")
        df["cpse"] = cpse.upper()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pick_duplicate_queries(df: pd.DataFrame, n: int = 5):
    """Re-derive cross-CPSE convergence groups (same logic as
    pipeline/run_normalize.py) and return n (query_text, expected_cpses)
    pairs -- the query is one CPSE's raw_description, expected_cpses is
    the set of OTHER CPSEs that should show up as high-similarity matches.
    """
    distinct_cpse_counts = df.groupby("normalized_description")["cpse"].nunique()
    cross_cpse_groups = distinct_cpse_counts[distinct_cpse_counts >= 2].sort_values(ascending=False)

    queries = []
    for norm_desc in cross_cpse_groups.index:
        if len(queries) >= n:
            break
        subset = df[df["normalized_description"] == norm_desc].drop_duplicates(subset="cpse")
        rows = subset.to_dict("records")
        query_row = rows[0]
        expected_cpses = {r["cpse"] for r in rows if r["cpse"] != query_row["cpse"]}
        queries.append((query_row["raw_description"], query_row["cpse"], expected_cpses))
    return queries


def print_matches(label, query_text, k=5):
    print(f"\nQUERY [{label}]: {query_text!r}")
    matches = find_similar(query_text, k=k)
    for rank, m in enumerate(matches, 1):
        print(f"  #{rank} sim={m['similarity']:.4f}  [{m['cpse']:4s}] {m['material_id']:10s} "
              f"raw={m['raw_description']!r}")
    return matches


def main():
    df = load_combined()
    print(f"Loaded {len(df)} combined rows across {df['cpse'].nunique()} CPSEs.")

    build_vector_store(df)
    print("Vector store built/updated at db/chroma (collection: materials).")

    print("\n" + "=" * 78)
    print("CROSS-CPSE DUPLICATE QUERIES (pulled from run_normalize.py convergence groups)")
    print("=" * 78)
    duplicate_queries = pick_duplicate_queries(df, n=5)
    for query_text, source_cpse, expected_cpses in duplicate_queries:
        label = f"{source_cpse} -> expect matches from {sorted(expected_cpses)}"
        matches = print_matches(label, query_text)
        matched_cpses = {m["cpse"] for m in matches if m["cpse"] != source_cpse}
        hit = expected_cpses & matched_cpses
        print(f"  >> expected CPSEs found in top-5: {sorted(hit)} / expected {sorted(expected_cpses)}")

    print("\n" + "=" * 78)
    print("TRUE-NEGATIVE QUERIES (CPSE-unique items -- should NOT match other CPSEs highly)")
    print("=" * 78)
    # Pull from the actual CPSE-exclusive item pool (generate_dataset.UNIQUE_SPLIT)
    # rather than inferring "uniqueness" from exact normalized-string grouping --
    # a shared item can also fail to string-converge (different word order) and
    # that is NOT a true negative, just a case embeddings should still catch.
    for cpse in ["IOCL", "ONGC"]:
        exclusive_texts = {t.upper() for t in UNIQUE_SPLIT[cpse]}
        candidates = df[(df["cpse"] == cpse) & (df["raw_description"].str.upper().isin(exclusive_texts))]
        row = candidates.iloc[0]
        matches = print_matches(f"{cpse} unique item", row["raw_description"])
        cross_cpse_hits = [m for m in matches if m["cpse"] != cpse]
        if cross_cpse_hits:
            top_cross = cross_cpse_hits[0]
            print(f"  >> highest-similarity match from a DIFFERENT CPSE: "
                  f"[{top_cross['cpse']}] sim={top_cross['similarity']:.4f} raw={top_cross['raw_description']!r}")
        else:
            print("  >> no matches at all from a different CPSE in top-k")


if __name__ == "__main__":
    main()

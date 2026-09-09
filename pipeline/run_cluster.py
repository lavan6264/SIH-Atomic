"""Cluster harmonized materials into Master Material Codes.

Reuses the embeddings + metadata already persisted in the Chroma
"materials" collection (built by pipeline/run_embed.py) -- no re-embedding.
"""

import numpy as np
import pandas as pd

from pipeline.embed import get_collection
from pipeline.cluster import assign_master_codes, save_to_db


def load_from_chroma() -> pd.DataFrame:
    collection = get_collection()
    result = collection.get(include=["embeddings", "metadatas"])

    rows = []
    for material_id, embedding, metadata in zip(result["ids"], result["embeddings"], result["metadatas"]):
        rows.append({
            "material_id": material_id,
            "cpse": metadata.get("cpse"),
            "raw_description": metadata.get("raw_description"),
            "normalized_description": metadata.get("normalized_description"),
            "quantity": metadata.get("quantity"),
            "unit_price": metadata.get("unit_price"),
            "embedding": np.asarray(embedding),
        })
    return pd.DataFrame(rows)


def main():
    df = load_from_chroma()
    print(f"Loaded {len(df)} items with embeddings from Chroma (no re-embedding).")

    result_df = assign_master_codes(df)
    save_to_db(result_df)
    print(f"Saved {len(result_df)} rows to db/atomic.sqlite (table: harmonized_materials).")

    total_clusters = result_df["master_material_code"].nunique()
    cluster_cpse_counts = result_df.groupby("master_material_code")["cpse"].nunique()
    multi_cpse_clusters = cluster_cpse_counts[cluster_cpse_counts >= 2].sort_values(ascending=False)
    needs_review_count = int(result_df["needs_review"].sum())
    avg_confidence = result_df["confidence"].mean()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Total items:                    {len(result_df)}")
    print(f"Total clusters (MMCs) formed:   {total_clusters}")
    print(f"Clusters spanning 2+ CPSEs:     {len(multi_cpse_clusters)}   <-- harmonization win count")
    print(f"Items flagged needs_review:     {needs_review_count} ({needs_review_count / len(result_df):.1%})")
    print(f"Average confidence:             {avg_confidence:.4f}")

    print("\nClusters (MMCs) formed per category:")
    print(result_df.groupby("category")["master_material_code"].nunique().sort_values(ascending=False).to_string())

    print("\n" + "=" * 78)
    print("5 EXAMPLE MULTI-CPSE CLUSTERS")
    print("=" * 78)
    for code in multi_cpse_clusters.index[:5]:
        cluster_rows = result_df[result_df["master_material_code"] == code]
        cpses = sorted(cluster_rows["cpse"].unique())
        print(f"\n{code}  ({len(cluster_rows)} items across {cpses})")
        for cpse in cpses:
            sample = cluster_rows[cluster_rows["cpse"] == cpse].iloc[0]
            print(f"  [{cpse:4s}] {sample['raw_description']!r}  (confidence={sample['confidence']:.4f})")


if __name__ == "__main__":
    main()

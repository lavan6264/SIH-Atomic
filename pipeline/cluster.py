"""scikit-learn agglomerative clustering into Master Material Codes.

Clustering only ever happens within a coarse category (get_category), never
across -- a valve and a bolt with a coincidentally similar embedding must
never end up in the same Master Material Code. Category keywords are
matched with regex word boundaries, not plain substring checks: a naive
`"tee" in text` would silently match inside "sTEEl"/"sTainless STEEl" and
miscategorize nearly every material in this dataset as a fitting.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "atomic.sqlite"

NEEDS_REVIEW_THRESHOLD = 0.85

# Checked in order; first match wins. Built from the actual vocabulary
# produced by pipeline/normalize.py on the synthetic dataset (valve/pipe/
# flange/gasket/pump/bolt names are always spelled out; "coupling" has no
# category of its own in the spec, so it's treated as a fitting).
_CATEGORY_PATTERNS = [
    ("valve", r"\bvalve\b"),
    ("pipe", r"\bpipe\b"),
    ("flange", r"\bflange\b"),
    ("gasket", r"\bgasket\b"),
    ("seal", r"\bseal\b|\bo[- ]?ring\b"),
    ("pump", r"\bpump\b"),
    ("bolt", r"\bbolt\b|\bnut\b"),
    ("fitting", r"\bfitting\b|\belbow\b|\btee\b|\breducer\b|\bcoupling\b"),
    ("motor", r"\bmotor\b"),
    ("battery", r"\bbattery\b"),
    ("wire_rope", r"\bwire\s*rope\b"),
]
_CATEGORY_PATTERNS = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _CATEGORY_PATTERNS]

# Hard lexical vetoes applied on top of embedding similarity, per category.
#
# Testing on this dataset found that all-MiniLM-L6-v2 does not reliably
# discriminate numeric attributes in short catalog-style text: e.g.
# "Ball Valve 4 Inch Carbon Steel" vs "Check Valve 4 Inch Carbon Steel"
# scores 0.87 cosine similarity, and "Gate Valve 2 Inch ... Class 150" vs
# "Gate Valve 4 Inch ... Class 300" scores 0.91 -- both clear the 0.85
# merge threshold, and both ranges overlap with genuine cross-CPSE
# paraphrase matches (which can score as low as ~0.95), so no single
# distance_threshold cleanly separates "same item, different wording" from
# "different item, similar wording". A ball valve is never a check valve
# and a 2 inch fitting is never a 4 inch one, regardless of embedding
# similarity, so for categories where an attribute is ALWAYS stated (never
# optional) across every CPSE style, we extract it and force a hard split
# between items whose value differs -- this only ever splits clusters the
# embedding model over-merged, never merges ones it kept apart.
#
# Pressure CLASS is deliberately excluded from the valve signature: it's
# genuinely optional there (~40-50% of rows omit it per CPSE style), so two
# rows of the identical valve -- one stating "Class 150", one silent on
# class -- must still merge; hard-requiring class to match would wrongly
# split them apart. Class is included for flange, where every CPSE style
# always states it.
_SIZE_RE = re.compile(
    r"\bM\d+\b"                  # metric threads: M16, M20, M24
    r"|\b\d+\s+\d+/\d+\s+Inch\b"  # mixed number: 1 1/2 Inch
    r"|\b\d+/\d+\s+Inch\b"       # fraction only: 3/4 Inch
    r"|\b\d+\s+Inch\b",          # whole number: 3 Inch
    re.IGNORECASE,
)
_FLOW_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:Cubic Meters Per Hour|M3/Hr)\b", re.IGNORECASE)
_MATERIAL_RE = re.compile(r"\bStainless Steel\s*\d*\b|\bCarbon Steel\b|\bGraphite\b", re.IGNORECASE)
_CLASS_RE = re.compile(r"\bClass\s*\d+\b", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bSch\d+\b", re.IGNORECASE)
_GRADE_RE = re.compile(r"\bGrade\s+[A-Za-z0-9.]+\b", re.IGNORECASE)
_VALVE_TYPE_RE = re.compile(r"\b(Ball|Gate|Globe|Check|Butterfly|Solenoid|Control)\b", re.IGNORECASE)
_FITTING_TYPE_RE = re.compile(r"\b(Elbow|Tee|Reducer|Coupling)\b", re.IGNORECASE)
_NUT_RE = re.compile(r"\bnut\b", re.IGNORECASE)
_STUD_RE = re.compile(r"\bstud\b", re.IGNORECASE)


def _find(pattern: "re.Pattern", text: str):
    match = pattern.search(text)
    return match.group(0).upper() if match else None


def _bolt_subtype(text: str) -> str:
    if _NUT_RE.search(text):
        return "nut"
    if _STUD_RE.search(text):
        return "stud"
    return "bolt"


def _flange_subtype(text: str):
    if re.search(r"\bBlind\b", text, re.IGNORECASE):
        return "blind"
    if re.search(r"\bWeld", text, re.IGNORECASE):
        return "weld_neck"
    return None


def _seal_subtype(text: str) -> str:
    return "oring" if re.search(r"\bo[- ]?ring\b", text, re.IGNORECASE) else "mechanical"


# Each function returns a tuple of hard-attribute values for one item's
# normalized_description; items in the same category only share a cluster
# if this tuple is identical (None entries mean "not applicable to this
# category," which is fine as long as it's uniformly None across the whole
# category -- true here since every category is checked with its own,
# always-present attribute set).
_HARD_DISCRIMINATORS = {
    "valve": lambda t: (_find(_VALVE_TYPE_RE, t), _find(_SIZE_RE, t), _find(_MATERIAL_RE, t)),
    "pipe": lambda t: (_find(_SIZE_RE, t), _find(_SCHEDULE_RE, t), _find(_MATERIAL_RE, t)),
    "flange": lambda t: (_flange_subtype(t), _find(_SIZE_RE, t), _find(_CLASS_RE, t), _find(_MATERIAL_RE, t)),
    "gasket": lambda t: (_find(_SIZE_RE, t), _find(_MATERIAL_RE, t)),
    "seal": lambda t: (_seal_subtype(t), _find(_SIZE_RE, t), _find(_MATERIAL_RE, t)),
    "pump": lambda t: (_find(_FLOW_RE, t), _find(_MATERIAL_RE, t)),
    "bolt": lambda t: (_bolt_subtype(t), _find(_SIZE_RE, t), _find(_GRADE_RE, t), _find(_MATERIAL_RE, t)),
    "fitting": lambda t: (_find(_FITTING_TYPE_RE, t), _find(_SIZE_RE, t), _find(_MATERIAL_RE, t)),
}


def get_category(normalized_description: str) -> str:
    """Coarsely bucket an item by keyword so clustering never crosses categories."""
    text = normalized_description or ""
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "other"


def cluster_category(items: list, embeddings: np.ndarray, similarity_threshold: float = 0.85) -> list:
    """Agglomerative-cluster one category's embeddings; return a label per item.

    items is accepted (per spec) but only its length matters here -- the
    actual grouping is purely a function of embeddings.
    """
    n = len(items)
    if n == 0:
        return []
    if n == 1:
        return [0]

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
    )
    labels = model.fit_predict(np.asarray(embeddings))
    return labels.tolist()


def _unit_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def assign_master_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket by category, cluster within each, assign Master Material Codes.

    Expects df to have material_id, cpse, raw_description,
    normalized_description, and embedding (one vector per row) columns.
    embedding is not required to be pre-normalized -- confidence is computed
    from unit-normalized copies internally. quantity/unit_price are carried
    through to the output when present in the input, but are not required.
    """
    df = df.copy()
    df["category"] = df["normalized_description"].apply(get_category)
    df["cluster_id"] = -1
    df["master_material_code"] = ""
    df["confidence"] = 0.0

    for category, group in df.groupby("category", sort=False):
        idx = group.index
        embeddings = np.stack(group["embedding"].to_numpy())
        items = group.to_dict("records")

        discriminator = _HARD_DISCRIMINATORS.get(category)
        if discriminator is not None:
            # Signature (type/size/material/...) is the primary grouping key,
            # not just a post-hoc veto: embeddings alone both over-merge
            # different items (see module docstring) AND under-merge the
            # same item when an optional attribute like valve class is
            # stated in one CPSE's row and omitted in another's, since that
            # wording difference alone can drop cosine similarity below the
            # AgglomerativeClustering threshold. The signature already
            # captures true physical identity for these categories, so
            # cluster_category()'s embedding-only clustering isn't used here.
            signatures = [discriminator(r["normalized_description"] or "") for r in items]
            remap = {sig: new_label for new_label, sig in enumerate(dict.fromkeys(signatures))}
            labels_arr = np.asarray([remap[sig] for sig in signatures])
        else:
            # No reliable hard signature for this category (other/motor/
            # battery/wire_rope) -- fall back to pure embedding clustering.
            labels = cluster_category(items, embeddings, similarity_threshold=0.85)
            labels_arr = np.asarray(labels)

        df.loc[idx, "cluster_id"] = labels_arr

        unit_embeddings = _unit_normalize(embeddings)

        for cluster_num, cluster_label in enumerate(sorted(set(labels_arr.tolist())), start=1):
            code = f"MMC-{category.upper()}-{cluster_num:04d}"
            mask = labels_arr == cluster_label
            cluster_vectors = unit_embeddings[mask]

            centroid = cluster_vectors.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            similarities = cluster_vectors @ centroid
            cluster_positions = idx[mask]
            df.loc[cluster_positions, "confidence"] = similarities
            df.loc[cluster_positions, "master_material_code"] = code

    df["needs_review"] = df["confidence"] < NEEDS_REVIEW_THRESHOLD

    output_cols = [
        "material_id", "cpse", "raw_description", "normalized_description",
        "category", "cluster_id", "master_material_code", "confidence", "needs_review",
    ]
    # quantity/unit_price are optional -- older Chroma collections built
    # before pricing was tracked won't have them, so don't require them.
    output_cols += [c for c in ("quantity", "unit_price") if c in df.columns]
    return df[output_cols]


def save_to_db(df: pd.DataFrame) -> None:
    """Persist the harmonized dataframe to db/atomic.sqlite, replacing the table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("harmonized_materials", conn, if_exists="replace", index=False)
    finally:
        conn.close()

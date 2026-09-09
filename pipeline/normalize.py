"""Regex + abbreviation-dictionary cleaning for fragmented ERP material text.

normalize_description() is a text-cleaning stage, not an entity parser: it
standardizes symbols, abbreviations, whitespace, and number/unit formatting
so that the SAME underlying item collapses to the same vocabulary across
CPSEs. It deliberately does not reorder words (e.g. IOCL's "type-first" vs
ONGC's "size-first" convention) -- resolving word-order and true semantic
equivalence is the job of the embedding + clustering stages downstream.

Processing is done on an internal uppercase working copy (so abbreviation
lookups and glued-code splitting are simple and case-insensitive), and
Title Case is applied once at the very end -- this is what actually
delivers "uppercase-normalize then title-case for consistency": if Title
Case were applied only up front, the symbol/abbreviation substitutions
injected afterward would reintroduce mixed casing and undo that
consistency.
"""

import re

from pipeline.abbreviation_dict import ABBREVIATIONS

# Abbreviation dict normalized to uppercase keys/values for matching against
# the uppercase working copy used throughout this function.
_ABBR = {k.strip().upper(): v.strip().upper() for k, v in ABBREVIATIONS.items()}

# Literal symbol -> word replacements (padded with spaces so they don't glue
# onto neighboring text, e.g. 3" -> "3 INCH " not "3INCH").
_SYMBOL_MAP = (
    ('"', " INCH "),
    ("'", " FT "),
    ("&", " AND "),
)

# Separators that should become plain whitespace before tokenizing, so
# "VLV-BALL-3IN-SS316" and "Ball Vlv, 3, SS-316" tokenize the same way.
_SEPARATOR_RE = re.compile(r"[,\-]")

# Glued codes that need splitting into isolated tokens before whole-word
# abbreviation matching can see them, e.g. "SS316" -> "SS 316" (so "SS"
# can be looked up), "CL150" -> "CL 150", "GR8.8"/"GRB7" -> "GR 8.8"/"GR B7".
_GLUED_PATTERNS = (
    (re.compile(r"\b(SS|CS)(\d{3})\b"), r"\1 \2"),
    (re.compile(r"\bCL(\d+)\b"), r"CL \1"),
    (re.compile(r"\bGR([0-9]+(?:\.[0-9]+)?|B[0-9]+M?)\b"), r"GR \1"),
    (re.compile(r"\b(\d+)(M3/?HR|M3H)\b"), r"\1 \2"),
)

# Number + unit standardization: "3IN", "3 IN", "3-IN" (hyphen already
# turned to a space by the time this runs) all collapse to "3 INCH".
_NUMBER_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?(?:/\d+)?)\s*(IN|INCH|FT)\b")


def _standardize_unit(match: "re.Match") -> str:
    number, unit = match.groups()
    unit = "INCH" if unit in ("IN", "INCH") else unit
    return f"{number} {unit}"


def normalize_description(raw_text: str) -> str:
    """Clean a single raw ERP material description into a standardized form.

    Order of operations (see module docstring for why Title Case is applied
    at the end rather than only at the start):
      1. Uppercase-normalize (casing consistency for all matching below)
      2. Replace symbols: " -> INCH, ' -> FT, & -> AND
      3. Expand abbreviations via ABBREVIATIONS, whole-token only
      4. Normalize whitespace
      5. Standardize number+unit formatting (3IN / 3 IN / 3-IN -> "3 INCH")
      6. Title-case the final result for consistency
    """
    if raw_text is None:
        return ""

    text = str(raw_text).strip().upper()
    if not text:
        return ""

    # 2. symbols -> words
    for sym, repl in _SYMBOL_MAP:
        text = text.replace(sym, repl)

    # 4. whitespace + separator normalization (commas/hyphens -> spaces)
    text = _SEPARATOR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # split glued codes ("SS316", "CL150", "GR8.8") into isolated tokens
    for pattern, repl in _GLUED_PATTERNS:
        text = pattern.sub(repl, text)

    # 5. number/unit formatting
    text = _NUMBER_UNIT_RE.sub(_standardize_unit, text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. whole-token abbreviation expansion (never a substring match, so
    # "IN" never touches tokens like "INSERTION")
    tokens = text.split(" ")
    expanded = [_ABBR.get(tok, tok) for tok in tokens]
    text = " ".join(expanded)

    # 4. final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    # 1./6. title-case for a consistent, human-readable final form
    return text.title()


if __name__ == "__main__":
    # Inline sanity checks (no test framework -- just asserts).
    # These are hand-picked variants of "3 inch ball valve" matching the
    # real conventions used across IOCL / ONGC / HPCL / BPCL.
    iocl_variant = "VLV-BALL-3IN-SS316"
    ongc_variant = "3 INCH BALL VALVE STAINLESS STEEL 316"
    hpcl_variant = 'Ball Vlv, 3", SS-316'
    bpcl_variant = "BALL VALVE 3IN SS316 CL150"  # has an extra pressure class

    n_iocl = normalize_description(iocl_variant)
    n_ongc = normalize_description(ongc_variant)
    n_hpcl = normalize_description(hpcl_variant)
    n_bpcl = normalize_description(bpcl_variant)

    print("IOCL:", iocl_variant, "->", n_iocl)
    print("ONGC:", ongc_variant, "->", n_ongc)
    print("HPCL:", hpcl_variant, "->", n_hpcl)
    print("BPCL:", bpcl_variant, "->", n_bpcl)

    # normalize.py is a text-cleaning stage, not a word-order-invariant
    # parser -- IOCL's "type-first" and ONGC's "size-first" conventions are
    # still different word orders after cleaning. Convergence at this stage
    # means "same normalized vocabulary" (checked as a token set), not
    # byte-identical strings; resolving word order is the embedding stage's
    # job.
    core_tokens = set(n_iocl.lower().split())
    assert core_tokens == set(n_ongc.lower().split()), (n_iocl, n_ongc)
    assert core_tokens == set(n_hpcl.lower().split()), (n_iocl, n_hpcl)

    # BPCL carries extra pressure-class info ("CL150" -> "Class 150"), so it
    # should be a superset of the core tokens, not an exact match.
    bpcl_tokens = set(n_bpcl.lower().split())
    assert core_tokens.issubset(bpcl_tokens), (core_tokens, bpcl_tokens)

    # spot-check the specific requirement-5 example: 3IN / 3 IN / 3-IN all
    # normalize to the same "3 inch" form.
    assert normalize_description("VLV-BALL-3IN-SS316") == normalize_description("VLV BALL 3 IN SS316")
    assert normalize_description("VLV-BALL-3IN-SS316") == normalize_description("VLV-BALL-3-IN-SS316")

    # symbol replacement sanity check
    assert normalize_description('PIPE 6" CS').split() == normalize_description("PIPE 6 INCH CS").split()

    print("\nAll inline asserts passed.")

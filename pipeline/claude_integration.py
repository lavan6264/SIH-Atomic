"""Claude API integration: match explanations, standard suggestions, NL search.

Every function wraps its API call in try/except and returns a clear, typed
fallback on any failure (missing key, network error, bad response) so a
flaky connection never takes down the dashboard or portal demo.
"""

import json
import os
import re

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"

# Small fixed reference list of real ISO/API/ASME standards covering the
# categories in this dataset (valves, flanges, pipe, bolts, gaskets).
STANDARDS_REFERENCE = {
    "ASME B16.5": "Pipe flanges and flanged fittings, NPS 1/2 through NPS 24",
    "ASME B16.47": "Large diameter steel flanges, NPS 26 through NPS 60",
    "ASME B16.34": "Valves - flanged, threaded, and welding end",
    "ASME B16.9": "Factory-made wrought steel buttwelding fittings (elbows, tees, reducers)",
    "ASME B16.11": "Forged steel socket-welding and threaded fittings",
    "ASME B16.20": "Metallic gaskets for pipe flanges - spiral wound, ring joint",
    "ASME B16.21": "Nonmetallic flat gaskets for pipe flanges",
    "ASME B18.2.1": "Square and hex bolts and screws",
    "ASME B18.2.2": "Square and hex nuts",
    "ASME B31.3": "Process piping design and construction",
    "API 6D": "Specification for pipeline valves (ball, gate, check, plug)",
    "API 600": "Steel gate valves for petroleum and natural gas industries",
    "API 602": "Compact steel gate valves for the petroleum industry",
    "API 594": "Wafer and dual-plate check valves",
    "API 609": "Butterfly valves",
    "API 5L": "Line pipe specification",
    "ISO 4014": "Hexagon head bolts, product grade A/B",
    "ISO 4032": "Hexagon nuts, style 1",
    "ISO 7005-1": "Metallic flanges - steel flanges",
    "MSS SP-44": "Steel pipeline flanges",
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a response, tolerating
    markdown code fences or stray prose around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {text!r}")
    return json.loads(match.group(0))


def explain_match(items: list) -> str:
    """Ask Claude why these cross-CPSE item descriptions are the same physical item.

    items: list of {"raw_description": ..., "cpse": ...} dicts clustered
    together under one Master Material Code.
    """
    if not items:
        return "No items to explain."

    listing = "\n".join(f"- [{item.get('cpse', '?')}] {item.get('raw_description', '')}" for item in items)
    prompt = (
        "These item descriptions from different companies' ERP systems were "
        "clustered together as the same physical material:\n\n"
        f"{listing}\n\n"
        "In 1-2 plain-English sentences, explain why these describe the same "
        "item despite the different wording (e.g. abbreviation style, word "
        "order, symbol usage). Be specific about the item itself (type, size, "
        "material). Do not use markdown formatting."
    )

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or "Explanation unavailable."
    except Exception as exc:
        return f"Explanation unavailable (Claude API error: {exc})."


def suggest_standard(normalized_description: str, category: str) -> dict:
    """Ask Claude to suggest the most likely matching standard from STANDARDS_REFERENCE."""
    fallback = {"standard": "Unknown", "reasoning": "Could not determine a standard."}

    reference_listing = "\n".join(f"- {code}: {desc}" for code, desc in STANDARDS_REFERENCE.items())
    prompt = (
        f"Item: \"{normalized_description}\" (category: {category})\n\n"
        f"Reference standards:\n{reference_listing}\n\n"
        "Which single standard from the list above most likely applies to this "
        "item? If none reasonably applies (e.g. it's not a valve/flange/pipe/"
        "bolt/gasket item), respond with \"standard\": \"None\".\n\n"
        "Respond with ONLY a JSON object, no other text, in exactly this form:\n"
        '{"standard": "<code from the list, or None>", "reasoning": "<one sentence>"}'
    )

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        result = _extract_json(text)
        if "standard" not in result or "reasoning" not in result:
            return fallback
        return {"standard": str(result["standard"]), "reasoning": str(result["reasoning"])}
    except Exception as exc:
        return {"standard": "Unknown", "reasoning": f"Could not determine a standard (Claude API error: {exc})."}


def _keyword_fallback_search(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Simple case-insensitive keyword search used when Claude's filter can't be parsed."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(w) > 2]
    if not words:
        return df.iloc[0:0]

    haystack = (df["raw_description"].fillna("") + " " + df["normalized_description"].fillna("")).str.lower()
    mask = pd.Series(False, index=df.index)
    for word in words:
        mask |= haystack.str.contains(re.escape(word), regex=True)
    return df[mask]


def _extract_size_inches(text: str):
    match = re.search(r"\b(\d+(?:\.\d+)?)\s+Inch\b", text or "", re.IGNORECASE)
    return float(match.group(1)) if match else None


def natural_language_search(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Translate a plain-English query into a structured filter and apply it to df.

    Falls back to a simple keyword search over raw_description /
    normalized_description if Claude's response can't be parsed into a
    valid filter, or if the API call fails outright.
    """
    if df.empty:
        return df

    known_categories = sorted(df["category"].dropna().unique()) if "category" in df.columns else []
    prompt = (
        f"Query: \"{query}\"\n\n"
        f"Known item categories: {', '.join(known_categories) if known_categories else 'unknown'}\n\n"
        "Translate this into a structured filter for a materials table with "
        "columns: category, raw_description, normalized_description (text "
        "descriptions include a size like '3 Inch'). Respond with ONLY a JSON "
        "object, no other text, in exactly this form (use null for any field "
        "that doesn't apply to the query):\n"
        '{"category": "<one of the known categories, or null>", '
        '"max_size_inches": <number or null>, "min_size_inches": <number or null>, '
        '"keyword": "<a single relevant keyword from the query, or null>"}'
    )

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        filter_spec = _extract_json(text)
    except Exception:
        return _keyword_fallback_search(query, df)

    if not isinstance(filter_spec, dict):
        return _keyword_fallback_search(query, df)

    try:
        result = df
        category = filter_spec.get("category")
        if category:
            result = result[result["category"].str.lower() == str(category).lower()]

        max_size = filter_spec.get("max_size_inches")
        min_size = filter_spec.get("min_size_inches")
        if max_size is not None:
            sizes = pd.to_numeric(result["normalized_description"].apply(_extract_size_inches))
            result = result[sizes <= float(max_size)]
        if min_size is not None:
            sizes = pd.to_numeric(result["normalized_description"].apply(_extract_size_inches))
            result = result[sizes >= float(min_size)]

        keyword = filter_spec.get("keyword")
        if keyword:
            haystack = (result["raw_description"].fillna("") + " " + result["normalized_description"].fillna("")).str.lower()
            result = result[haystack.str.contains(re.escape(str(keyword).lower()), regex=True)]

        return result
    except Exception:
        return _keyword_fallback_search(query, df)

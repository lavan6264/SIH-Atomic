"""Synthetic 4-CPSE material master dataset generator.

Produces data/{iocl,ongc,hpcl,bpcl}_materials.csv (~150 rows each) that
deliberately fragments the SAME physical items across CPSEs using each
company's own ERP text conventions (casing, abbreviation, word order,
symbol usage). Also injects exact duplicates and CPSE-unique items
("true negatives") so downstream clustering has real signal to filter.

Reproducible: seeded via random.seed(SEED) / np.random.seed(SEED).
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROWS_PER_CPSE = 150
CPSES = ["IOCL", "ONGC", "HPCL", "BPCL"]

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Material representations per house style
# ---------------------------------------------------------------------------

MAT_STYLE = {
    "SS316": {"iocl": "SS316", "ongc": "STAINLESS STEEL 316", "hpcl": "SS-316", "bpcl": "SS316"},
    "SS304": {"iocl": "SS304", "ongc": "STAINLESS STEEL 304", "hpcl": "SS-304", "bpcl": "SS304"},
    "CS": {"iocl": "CS", "ongc": "CARBON STEEL", "hpcl": "CS", "bpcl": "CS"},
    "GRAPHITE": {"iocl": "GR", "ongc": "GRAPHITE", "hpcl": "Graphite", "bpcl": "GRAPHITE"},
}

VALVE_WORDS = {
    "BALL": {"iocl": "BALL", "ongc": "BALL", "hpcl": "Ball", "bpcl": "BALL"},
    "GATE": {"iocl": "GATE", "ongc": "GATE", "hpcl": "Gate", "bpcl": "GATE"},
    "GLOBE": {"iocl": "GLB", "ongc": "GLOBE", "hpcl": "Globe", "bpcl": "GLOBE"},
    "CHECK": {"iocl": "CHK", "ongc": "CHECK", "hpcl": "Check", "bpcl": "CHECK"},
    "BUTTERFLY": {"iocl": "BFLY", "ongc": "BUTTERFLY", "hpcl": "Butterfly", "bpcl": "BUTTERFLY"},
}

FRACTIONS = {0.5: "1/2", 0.75: "3/4", 1.25: "1-1/4", 1.5: "1-1/2", 2.5: "2-1/2"}


def size_str(size):
    if size in FRACTIONS:
        return FRACTIONS[size]
    if float(size).is_integer():
        return str(int(size))
    return str(size)


def jitter_case(s, rng):
    """Occasionally flip casing to simulate messy ERP entry."""
    r = rng.random()
    if r < 0.10:
        return s.upper()
    if r < 0.18:
        return s.lower()
    if r < 0.24:
        return s.title()
    return s


def jitter_space(s, rng):
    """Occasionally add/collapse whitespace around punctuation."""
    if rng.random() < 0.12:
        s = s.replace(", ", " ,").replace("  ", " ")
    if rng.random() < 0.10:
        s = s.replace("-", " - ")
    return s


# ---------------------------------------------------------------------------
# Category renderers: each returns the 4 CPSE-house-style strings for one
# canonical (shared) item, given a per-row rng for light template variation.
# ---------------------------------------------------------------------------

def render_valve(item, rng):
    vt = item["valve_type"]
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    cls = item.get("class_")

    iocl = f"VLV-{VALVE_WORDS[vt]['iocl']}-{sz}IN-{mat['iocl']}"
    if cls and rng.random() < 0.5:
        iocl += f"-CL{cls}"

    ongc_variants = [
        f"{sz} INCH {VALVE_WORDS[vt]['ongc']} VALVE {mat['ongc']}",
        f"{VALVE_WORDS[vt]['ongc']} VALVE {mat['ongc']} {sz} INCH",
    ]
    ongc = rng.choice(ongc_variants)
    if cls and rng.random() < 0.4:
        ongc += f" CLASS {cls}"

    hpcl = f"{VALVE_WORDS[vt]['hpcl']} Vlv, {sz}\", {mat['hpcl']}"
    if cls and rng.random() < 0.4:
        hpcl += f", Cl {cls}"

    bpcl = f"{VALVE_WORDS[vt]['bpcl']} VALVE {sz}IN {mat['bpcl']}"
    if cls:
        bpcl += f" CL{cls}"

    return iocl, ongc, hpcl, bpcl


def render_pipe(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    sch = item["schedule"]
    ptype = item["pipe_type"]  # SEAMLESS / WELDED
    ptype_ongc = {"SEAMLESS": "SEAMLESS", "WELDED": "WELDED"}[ptype]
    ptype_iocl = {"SEAMLESS": "SMLS", "WELDED": "WLD"}[ptype]

    iocl = f"PIPE-{sz}IN-{mat['iocl']}-{sch}-{ptype_iocl}"
    ongc = f"{sz} INCH {mat['ongc']} PIPE {sch} {ptype_ongc}"
    hpcl = f"Pipe, {sz}\", {mat['hpcl']}, {sch}"
    bpcl = f"PIPE {sz}IN {mat['bpcl']} {sch}"
    return iocl, ongc, hpcl, bpcl


def render_bolt(item, rng):
    sz = item["size"]  # already a code like "M16" or '3/4IN'
    mat = MAT_STYLE[item["material"]]
    grade = item["grade"]
    kind = item["kind"]  # HEX BOLT / STUD BOLT / HEX NUT

    if kind == "HEX BOLT":
        iocl = f"BOLT-HX-{sz}-{mat['iocl']}-GR{grade}"
        ongc = f"{sz} HEX BOLT {mat['ongc']} GRADE {grade}"
        hpcl = f"Hx Bolt, {sz}, {mat['hpcl']}, Gr {grade}"
        bpcl = f"HEX BOLT {sz} {mat['bpcl']} GR{grade}"
    elif kind == "STUD BOLT":
        iocl = f"BOLT-STUD-{sz}-{mat['iocl']}-GR{grade}"
        ongc = f"{sz} STUD BOLT {mat['ongc']} GRADE {grade}"
        hpcl = f"Stud Bolt, {sz}, {mat['hpcl']}, Gr {grade}"
        bpcl = f"STUD BOLT {sz} {mat['bpcl']} GR{grade}"
    else:  # HEX NUT
        iocl = f"NUT-HX-{sz}-{mat['iocl']}-GR{grade}"
        ongc = f"{sz} HEX NUT {mat['ongc']} GRADE {grade}"
        hpcl = f"Hx Nut, {sz}, {mat['hpcl']}, Gr {grade}"
        bpcl = f"HEX NUT {sz} {mat['bpcl']} GR{grade}"
    return iocl, ongc, hpcl, bpcl


def render_gasket(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    iocl = f"GSKT-SW-{sz}IN-{mat['iocl']}"
    ongc = f"{sz} INCH SPIRAL WOUND GASKET {mat['ongc']}"
    hpcl = f"Gskt, Spiral Wound, {sz}\", {mat['hpcl']}"
    bpcl = f"GASKET SW {sz}IN {mat['bpcl']}"
    return iocl, ongc, hpcl, bpcl


def render_flange(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    cls = item["class_"]
    ftype = item["flange_type"]  # WN / BLIND
    ftype_words = {"WN": "WELD NECK", "BLIND": "BLIND"}[ftype]
    ftype_hpcl = {"WN": "Weld Neck", "BLIND": "Blind"}[ftype]

    iocl = f"FLG-{ftype}-{sz}IN-{mat['iocl']}-CL{cls}"
    ongc = f"{sz} INCH {ftype_words} FLANGE {mat['ongc']} CLASS {cls}"
    hpcl = f"Flg, {ftype_hpcl}, {sz}\", {mat['hpcl']}, Cl {cls}"
    bpcl = f"FLANGE {ftype} {sz}IN {mat['bpcl']} CL{cls}"
    return iocl, ongc, hpcl, bpcl


def render_pump(item, rng):
    flow = item["flow"]
    head = item["head"]
    mat = MAT_STYLE[item["material"]]
    iocl = f"PUMP-CENTRF-{flow}M3H-{head}M-{mat['iocl']}"
    ongc = f"CENTRIFUGAL PUMP {flow} M3/HR {head}M HEAD {mat['ongc']}"
    hpcl = f"Pump, Centrifugal, {flow}m3/hr, {head}m, {mat['hpcl']}"
    bpcl = f"CENTRIFUGAL PUMP {flow}M3/HR {head}M {mat['bpcl']}"
    return iocl, ongc, hpcl, bpcl


def render_fitting(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    sub = item["subtype"]  # ELBOW / TEE / REDUCER
    words = {"ELBOW": ("ELB", "ELBOW 90DEG", "Elbow 90deg"),
             "TEE": ("TEE", "TEE", "Tee"),
             "REDUCER": ("RED", "REDUCER", "Reducer")}
    iocl_w, ongc_w, hpcl_w = words[sub]

    iocl = f"FIT-{iocl_w}-{sz}IN-{mat['iocl']}"
    ongc = f"{sz} INCH {ongc_w} {mat['ongc']}"
    hpcl = f"{hpcl_w}, {sz}\", {mat['hpcl']}"
    bpcl = f"{sub} {sz}IN {mat['bpcl']}"
    return iocl, ongc, hpcl, bpcl


def render_seal(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    sub = item["subtype"]  # ORING / MECH
    if sub == "ORING":
        iocl = f"SEAL-ORING-{sz}IN-{mat['iocl']}"
        ongc = f"{sz} INCH O-RING SEAL {mat['ongc']}"
        hpcl = f"O-Ring, {sz}\", {mat['hpcl']}"
        bpcl = f"ORING SEAL {sz}IN {mat['bpcl']}"
    else:
        iocl = f"SEAL-MECH-{sz}IN-{mat['iocl']}"
        ongc = f"MECHANICAL SEAL {sz} INCH {mat['ongc']}"
        hpcl = f"Mech Seal, {sz}\", {mat['hpcl']}"
        bpcl = f"MECHANICAL SEAL {sz}IN {mat['bpcl']}"
    return iocl, ongc, hpcl, bpcl


def render_coupling(item, rng):
    sz = size_str(item["size"])
    mat = MAT_STYLE[item["material"]]
    iocl = f"CPLG-{sz}IN-{mat['iocl']}"
    ongc = f"{sz} INCH COUPLING {mat['ongc']}"
    hpcl = f"Cplg, {sz}\", {mat['hpcl']}"
    bpcl = f"COUPLING {sz}IN {mat['bpcl']}"
    return iocl, ongc, hpcl, bpcl


RENDERERS = {
    "valve": render_valve,
    "pipe": render_pipe,
    "bolt": render_bolt,
    "gasket": render_gasket,
    "flange": render_flange,
    "pump": render_pump,
    "fitting": render_fitting,
    "seal": render_seal,
    "coupling": render_coupling,
}

# price/qty ranges per category, used for both shared and unique items
PRICE_RANGE = {
    "valve": (2000, 150000),
    "pipe": (200, 5000),
    "bolt": (5, 500),
    "gasket": (50, 2000),
    "flange": (1000, 50000),
    "pump": (50000, 500000),
    "fitting": (100, 5000),
    "seal": (200, 8000),
    "coupling": (500, 10000),
    "other": (50, 20000),
}
QTY_RANGE = {
    "valve": (1, 100),
    "pipe": (10, 1000),
    "bolt": (50, 2000),
    "gasket": (5, 300),
    "flange": (1, 100),
    "pump": (1, 5),
    "fitting": (5, 300),
    "seal": (2, 200),
    "coupling": (1, 100),
    "other": (1, 500),
}

# ---------------------------------------------------------------------------
# Canonical shared item catalog (the "same real item, 4 different spellings")
# ---------------------------------------------------------------------------

CANONICAL_ITEMS = []

for vt, sz, mat, cls in [
    ("BALL", 1, "SS316", 150), ("BALL", 2, "SS316", 150), ("BALL", 3, "SS316", 150),
    ("BALL", 4, "CS", 150), ("BALL", 6, "SS304", 300),
    ("GATE", 2, "CS", 150), ("GATE", 4, "CS", 300), ("GATE", 6, "SS316", 150),
    ("GLOBE", 1.5, "SS316", 150), ("GLOBE", 3, "CS", 300),
    ("CHECK", 2, "SS304", 150), ("CHECK", 4, "CS", 150),
    ("BUTTERFLY", 6, "CS", 150), ("BUTTERFLY", 8, "SS316", 150),
]:
    CANONICAL_ITEMS.append({"category": "valve", "valve_type": vt, "size": sz,
                             "material": mat, "class_": cls})

for sz, mat, sch, ptype in [
    (2, "CS", "SCH40", "SEAMLESS"), (3, "CS", "SCH40", "SEAMLESS"),
    (4, "SS316", "SCH80", "SEAMLESS"), (6, "CS", "SCH40", "WELDED"),
    (8, "CS", "SCH80", "WELDED"),
]:
    CANONICAL_ITEMS.append({"category": "pipe", "size": sz, "material": mat,
                             "schedule": sch, "pipe_type": ptype})

for kind, sz, mat, grade in [
    ("HEX BOLT", "M16", "CS", "8.8"), ("HEX BOLT", "M20", "SS304", "B8"),
    ("HEX BOLT", "3/4IN", "CS", "B7"), ("STUD BOLT", "M24", "CS", "B7"),
    ("STUD BOLT", "1IN", "SS316", "B8M"), ("HEX NUT", "M16", "CS", "8"),
]:
    CANONICAL_ITEMS.append({"category": "bolt", "kind": kind, "size": sz,
                             "material": mat, "grade": grade})

for sz, mat in [
    (2, "SS316"), (3, "SS316"), (4, "GRAPHITE"), (6, "SS316"), (8, "GRAPHITE"),
]:
    CANONICAL_ITEMS.append({"category": "gasket", "size": sz, "material": mat})

for sz, mat, cls, ftype in [
    (2, "CS", 150, "WN"), (3, "CS", 150, "WN"), (4, "SS316", 300, "WN"),
    (6, "CS", 150, "BLIND"), (8, "CS", 300, "BLIND"), (3, "SS304", 150, "BLIND"),
]:
    CANONICAL_ITEMS.append({"category": "flange", "size": sz, "material": mat,
                             "class_": cls, "flange_type": ftype})

for flow, head, mat in [
    (50, 30, "CS"), (100, 50, "SS316"), (200, 80, "CS"), (75, 40, "SS304"),
]:
    CANONICAL_ITEMS.append({"category": "pump", "flow": flow, "head": head, "material": mat})

for sub, sz, mat in [
    ("ELBOW", 2, "CS"), ("ELBOW", 4, "SS316"), ("TEE", 3, "CS"),
    ("TEE", 6, "SS316"), ("REDUCER", 4, "CS"), ("REDUCER", 6, "SS304"),
]:
    CANONICAL_ITEMS.append({"category": "fitting", "subtype": sub, "size": sz, "material": mat})

for sub, sz, mat in [
    ("ORING", 2, "SS316"), ("ORING", 4, "SS304"), ("MECH", 3, "SS316"), ("MECH", 6, "SS316"),
]:
    CANONICAL_ITEMS.append({"category": "seal", "subtype": sub, "size": sz, "material": mat})

for sz, mat in [(2, "CS"), (4, "SS316")]:
    CANONICAL_ITEMS.append({"category": "coupling", "size": sz, "material": mat})

# ---------------------------------------------------------------------------
# CPSE-unique "true negative" items: genuinely different, never fragmented
# across companies. Each CPSE draws its own disjoint subset.
# ---------------------------------------------------------------------------

UNIQUE_POOL = [
    "SAFETY HELMET YELLOW IS2925", "NITRILE GLOVES SIZE L BOX OF 100",
    "WELDING ROD E7018 3.2MM", "CABLE GLAND BRASS 20MM NPT",
    "JUNCTION BOX SS304 IP65", "FIRE EXTINGUISHER CO2 5KG",
    "GREASE NLGI-2 180KG DRUM", "V-BELT A-SECTION A52",
    "BALL BEARING 6205 2RS", "FILTER ELEMENT 10 MICRON PLEATED",
    "THERMOWELL SS316 6IN INSERTION", "PRESSURE GAUGE 0-100 BAR 4IN DIAL",
    "TEMPERATURE TRANSMITTER 4-20MA HART", "CHAIN SLING 2TON 3MTR",
    "WIRE ROPE 12MM GALVANIZED", "PAINT EPOXY ZINC RICH 20LTR",
    "INSULATION TAPE PVC BLACK 19MM", "CABLE TRAY PERFORATED 300MM GI",
    "TERMINAL BLOCK 4MM SQ GREY", "SAFETY GOGGLES CLEAR LENS",
    "RESPIRATOR CARTRIDGE ORGANIC VAPOUR", "HYDRAULIC HOSE 1IN 2WIRE BRAID",
    "AIR FILTER REGULATOR LUBRICATOR UNIT", "LEVEL TRANSMITTER RADAR TYPE",
    "SOLENOID VALVE 24VDC 1/2IN", "LIMIT SWITCH ROTARY TYPE",
    "BATTERY LEAD ACID 12V 100AH", "FIRE HOSE REEL 30MTR",
    "SPILL KIT OIL ABSORBENT 200LTR", "EARTHING STRIP COPPER 25X3MM",
    "MOTOR 3PH 415V 15KW TEFC", "GEARBOX HELICAL 10:1 RATIO",
    "STRAINER Y-TYPE 4IN CS", "SIGHT GLASS LEVEL INDICATOR 12IN",
    "DESICCANT BREATHER 2IN NPT", "ROTAMETER 0-50 LPM GLASS TUBE",
    "CONTROL VALVE POSITIONER PNEUMATIC", "CABLE 3C X 2.5SQMM ARMOURED",
    "LADDER FIBERGLASS 12FT", "TARPAULIN HDPE 6X9MTR",
]
random.shuffle(UNIQUE_POOL)
UNIQUE_SPLIT = {
    "IOCL": UNIQUE_POOL[0:10],
    "ONGC": UNIQUE_POOL[10:20],
    "HPCL": UNIQUE_POOL[20:30],
    "BPCL": UNIQUE_POOL[30:40],
}

# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

STYLE_KEY = {"IOCL": 0, "ONGC": 1, "HPCL": 2, "BPCL": 3}


def price_qty_for(category, rng_np):
    lo, hi = PRICE_RANGE.get(category, PRICE_RANGE["other"])
    qlo, qhi = QTY_RANGE.get(category, QTY_RANGE["other"])
    price = round(rng_np.uniform(lo, hi), 2)
    qty = int(rng_np.randint(qlo, qhi + 1))
    return qty, price


def build_cpse_rows(cpse, seq_start):
    rows = []
    py_rng = random.Random(SEED + STYLE_KEY[cpse])
    seq = seq_start

    n_shared = 100
    n_dupes = 15
    n_unique = ROWS_PER_CPSE - n_shared - n_dupes  # 35

    # 1) shared canonical items, sampled with repetition, house-style rendered
    shared_texts = []
    for _ in range(n_shared):
        item = py_rng.choice(CANONICAL_ITEMS)
        renderer = RENDERERS[item["category"]]
        texts = renderer(item, py_rng)  # (iocl, ongc, hpcl, bpcl)
        raw = texts[STYLE_KEY[cpse]]
        raw = jitter_case(raw, py_rng)
        raw = jitter_space(raw, py_rng)
        shared_texts.append(raw)
        qty, price = price_qty_for(item["category"], np.random)
        rows.append([f"{cpse}-{seq:05d}", raw, qty, price])
        seq += 1

    # 2) exact duplicates of already-generated shared rows (same wording,
    #    new line-item id/qty/price, as a real ERP re-order would look)
    for _ in range(n_dupes):
        raw = py_rng.choice(shared_texts)
        qty, price = price_qty_for("other", np.random)
        rows.append([f"{cpse}-{seq:05d}", raw, qty, price])
        seq += 1

    # 3) CPSE-unique items: genuinely different, no cross-CPSE match (true negatives)
    unique_pool = UNIQUE_SPLIT[cpse]
    for _ in range(n_unique):
        base = py_rng.choice(unique_pool)
        raw = jitter_case(base, py_rng)
        qty, price = price_qty_for("other", np.random)
        rows.append([f"{cpse}-{seq:05d}", raw, qty, price])
        seq += 1

    py_rng.shuffle(rows)
    return rows, seq


def main():
    all_counts = {}
    samples = {}
    for cpse in CPSES:
        rows, _ = build_cpse_rows(cpse, seq_start=1)
        df = pd.DataFrame(rows, columns=["material_id", "raw_description", "quantity", "unit_price"])
        out_path = OUT_DIR / f"{cpse.lower()}_materials.csv"
        df.to_csv(out_path, index=False)
        all_counts[cpse] = len(df)
        samples[cpse] = df.sample(n=min(5, len(df)), random_state=SEED)

    print("Row counts per CPSE:")
    for cpse, count in all_counts.items():
        print(f"  {cpse}: {count}")
    print(f"  TOTAL: {sum(all_counts.values())}")

    print("\nSample rows:")
    for cpse in CPSES:
        print(f"\n--- {cpse} ---")
        print(samples[cpse].to_string(index=False))


if __name__ == "__main__":
    main()

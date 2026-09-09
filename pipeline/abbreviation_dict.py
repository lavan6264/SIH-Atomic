# Common industrial/ERP shorthand -> standardized term.
# Keys are matched case-insensitively by normalize.py against tokens/symbols
# found in raw_description strings from the 4-CPSE synthetic dataset.

ABBREVIATIONS = {
    # --- required baseline ---
    "VLV": "valve",
    "PIPE": "pipe",
    "CS": "carbon steel",
    "SS": "stainless steel",
    "IN": "inch",
    "INCH": "inch",
    '"': "inch",
    "HX": "hex",
    "BOLT": "bolt",
    "GSKT": "gasket",
    "FLG": "flange",
    "CL": "class",
    "NPT": "NPT thread",

    # --- valve types ---
    "BALL": "ball",
    "GATE": "gate",
    "GLB": "globe",
    "GLOBE": "globe",
    "CHK": "check",
    "CHECK": "check",
    "BFLY": "butterfly",
    "BTRFLY": "butterfly",

    # --- pipe / fittings ---
    "SMLS": "seamless",
    "SEAMLESS": "seamless",
    "WLD": "welded",
    "WELD": "welded",
    "SCH": "schedule",
    "FIT": "fitting",
    "ELB": "elbow",
    "ELBOW": "elbow",
    "TEE": "tee",
    "RED": "reducer",
    "RDCR": "reducer",
    "CPLG": "coupling",
    "NB": "nominal bore",
    "OD": "outer diameter",
    "ID": "inner diameter",
    "THD": "thread",
    "DIA": "diameter",

    # --- bolts / fasteners ---
    "STUD": "stud",
    "NUT": "nut",
    "GR": "grade",
    "WT": "wall thickness",

    # --- gasket / flange ---
    "SW": "spiral wound",
    "WN": "weld neck",
    "BLND": "blind",
    "BLIND": "blind",

    # --- materials ---
    "SS316": "stainless steel 316",
    "SS304": "stainless steel 304",
    "GR.": "graphite",
    "GRPH": "graphite",

    # --- pump ---
    "PMP": "pump",
    "CENTRF": "centrifugal",
    "CENT": "centrifugal",
    "M3H": "cubic meters per hour",
    "M3/HR": "cubic meters per hour",
    "HD": "head",

    # --- seals ---
    "MECH": "mechanical",
    "ORING": "o-ring",
    "O-RING": "o-ring",

    # --- standards / misc ---
    "ANSI": "ANSI standard",
    "API": "API standard",
    "STD": "standard",
    "EA": "each",
    "QTY": "quantity",
}

# Atomic — Material Code Harmonization (SIH26099)

Hackathon prototype: an AI-driven material code harmonization system for
CPSEs (IOCL, ONGC, HPCL, BPCL). Four-layer pipeline: ingest & clean
fragmented ERP text → semantic embedding → clustering into a Master
Material Code → serve via a human-review dashboard and cross-CPSE
search portal.

## Status

This is a fresh scaffold, not a working implementation yet. Every module
listed below is a placeholder (a one-line comment stating its intended
responsibility) except `app/main.py`, which is a working Streamlit
entrypoint with the title and two empty tabs — enough to confirm
`streamlit run app/main.py` runs. No pipeline logic, no data, no DB
schema, no Chroma collection exist yet.

## Structure

- `data/` — synthetic 4-CPSE material master CSVs (not yet created)
- `pipeline/normalize.py` — regex + abbreviation-dictionary cleaning (placeholder)
- `pipeline/embed.py` — sentence-transformers embedding logic (placeholder)
- `pipeline/cluster.py` — scikit-learn agglomerative clustering (placeholder)
- `app/main.py` — Streamlit entrypoint, routes between the two pages below (working placeholder)
- `app/dashboard.py` — human-in-the-loop review UI (placeholder)
- `app/portal.py` — cross-CPSE search portal (placeholder)
- `db/` — SQLite database file lives here (not yet created)
- `notebooks/` — Jupyter notebooks for interactive embedding/clustering tuning
- `assets/` — generated charts/diagrams
- `venv/` — Python virtual environment (gitignored)

## Environment

- Python 3.13 (official python.org/Windows build via the `py` launcher —
  **do not** recreate the venv with whatever `python` resolves to on
  PATH first; on this machine that's an MSYS2 UCRT64 build that produces
  a POSIX-style `venv/bin/` layout instead of `venv/Scripts/` and risks
  ABI mismatches with compiled wheels like torch/scikit-learn/chromadb's
  onnxruntime. Use `py -3.13 -m venv venv` if it ever needs recreating.)
- Activate: `.\venv\Scripts\Activate.ps1` (PowerShell) or
  `venv/Scripts/activate` (bash/Git Bash).
- Dependencies: streamlit, sentence-transformers (pulls in torch),
  chromadb, scikit-learn, pandas, numpy, anthropic, jupyter — installed
  and verified (all imports smoke-tested clean); see `requirements.txt`
  for the full pinned list (186 packages incl. transitive deps).
- Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`. `.env`
  is gitignored; never commit real keys.

## Running

`streamlit run app/main.py` from the project root with the venv active.

## Conventions

- Keep pipeline stages (`normalize.py` → `embed.py` → `cluster.py`)
  independent, callable modules — the dashboard and portal apps import
  from `pipeline/`, not the other way around.
- SQLite (`db/`) is the source of truth for reviewed/confirmed Master
  Material Code mappings; Chroma is the vector index over embeddings,
  treated as a rebuildable cache, not source of truth.

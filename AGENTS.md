# Repository Guidelines

## Project Structure & Module Organization
- `transcription_pipeline/` is the installable package. Key submodules: `preprocessing/` (data import + zarr/tiff export), `spot_analysis/` and `spot_pipeline.py` (spot detection and quantification), `tracking/` and `step_detection/` (trajectory handling), `nuclear_analysis/` (per nucleus metrics), `utils/` (shared math/image helpers), and `gui/` for the Napari-based QA widget.
- The notebooks in the repo root document manual workflows (e.g., `Modify_metadata.ipynb`), while `Data/` stores example Garcia Lab datasets following the documented folder hierarchy.
- Documentation sources live in `transcription_pipeline/docs/source/`; built artefacts are ignored except for the local `build/` output used for review.

## Build, Test, and Development Commands
- `mamba env create -f environment.yml` (or `conda`) provisions a full scientific stack; alternatively install base requirements via `pip install -e .` to work in editable mode.
- `pip install -e . && install_bioformats_jar` keeps the package editable and downloads the `loci_tools.jar` required by Bio-Formats readers.
- `python -m pip install -r requirements.txt` refreshes runtime dependencies in lightweight environments.
- `make html -C transcription_pipeline/docs` renders the Sphinx docs to `transcription_pipeline/docs/build/html` for validation before publishing.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, line length ≤ 88 chars, and type-aware docstrings matching existing modules (`:param`, `:return`).
- Prefer descriptive snake_case for functions and variables, UpperCamelCase for classes, and reserve ALL_CAPS for module constants such as `DEFAULT_QUANTIFICATION_COLUMN`.
- Use vectorised NumPy/Pandas operations where practical; align new utilities with patterns in `transcription_pipeline/utils/`.

## Testing Guidelines
- There is no automated suite yet; add `pytest`-compatible tests alongside new code under `tests/` or the relevant package subdirectory.
- Exercise pipeline entrypoints with representative sample folders under `Data/`, but keep fixtures small and anonymised.
- Run `pytest` locally before pushing; include regression notebooks or screenshots only when a UI change cannot be automated.

## Commit & Pull Request Guidelines
- Recent history mixes descriptive and narrative commits; standardise on imperative, present-tense messages (`"Add spot QC widget shortcuts"`).
- Group related changes, reference issues as `#123`, and document breaking behaviour in the body.
- Pull requests should describe expected inputs/outputs, list test commands executed, and include screenshots/GIFs when GUI elements or Napari overlays change.

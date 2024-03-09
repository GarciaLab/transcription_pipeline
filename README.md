# Transcription spot analysis pipeline
Python pipeline for transcription spot tracking and analysis.

# Installation instructions
1. (Optional) Create a mamba/conda virtualenv from the `environment.yml` file.
2. Run `pip install . && install_bioformats_jar`. If you want to keep editing the source code without having to build and reinstall between edits, use `pip install -e . && install_bioformats_jar` instead - this will also mean that any changed pulled from Github will instantaneously take effect.

Skipping step 1 should still yield a functional install of the `transcription_pipeline` package, but will not include convenient tools included in the conda/mamba environment file such as JupyterLab and tools for interactive plots.

## Data folder structure
For individual datasets, we keep consistent with the Garcia Lab convention
for folder structure (Date Folder > Name Folder > Data File + FullEmbryo
Folder, FullEmbryo Folder > Mid, Surf).

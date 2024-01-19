# Transcription spot analysis pipeline
Python pipeline for transcription spot tracking and analysis.

# Installation instructions
1. (Optional) Create a mamba/conda virtualenv from the `environment.yml` file.
2. Run `pip install . && install_bioformats_jar`.

Skipping step 1 should still yield a functional install of the `transcription_pipeline` package, but will not include convenient tools included in the conda/mamba environment file such as JupyterLab and tools for interactive plots.
If you would like to modify the pipeline files without needed to rebuild and reinstall through pip, you can instead use `pip install -e . && install_bioformats_jar`.

## Data folder structure
For individual datasets, we keep consistent with the Garcia Lab convention
for folder structure (Date Folder > Name Folder > Data File + FullEmbryo
Folder, FullEmbryo Folder > Mid, Surf).

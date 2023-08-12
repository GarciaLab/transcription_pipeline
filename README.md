# Transcription spot analysis pipeline
Python pipeline for transcription spot tracking and analysis.

# Installation instructions
1. (Optional) Create a mamba/conda virtualenv from the `environment.yml` file.
2. If you did a clean pull from the repository and created a mamba/conda environment from the `environment.yml` file, the `transcription_package` pipeline is already installed, so you can just run `install_bioformats_jar` to set up the Bioformats javabridge. Otherwise, run `pip install . && install_bioformats_jar`.

Skipping step 1 should still yield a functional install of the `transcription_pipeline` package, but will not include convenient tools included in the conda/mamba environment file such as JupyterLab and tools for interactive plots.

## Data folder structure
For individual datasets, we keep consistent with the Garcia Lab convention
for folder structure (Date Folder > Name Folder > Data File + FullEmbryo
Folder, FullEmbryo Folder > Mid, Surf).

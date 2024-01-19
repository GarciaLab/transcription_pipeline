import jdk
import sys
import os
import shutil
from pathlib import Path
import warnings
import pims
import inspect
import transcription_pipeline


def install_bioformats():
    """
    Installs the specific version of loci.tools required for the PIMS Bioformats
    reader.
    """
    environment_path = Path(sys.prefix)
    pims_path = Path(inspect.getfile(pims)).parents[0]
    loci_tools_path = Path(inspect.getfile(transcription_pipeline)).parents[0] / "loci_tools.jar"

    try:
        shutil.copy(loci_tools_path, pims_path / "loci_tools.jar")
    except OSError:
        warnings.warn(
            "`loci_tools.jar` not found in package folder."
        )
        
    path_parents = []
    for parent in pims_path.parents:
        if parent.name == "lib":
            path_parents.append(parent)

    try:
        jdk_path = path_parents[0]
    except IndexError:
        raise Exception("Could not find target folder for JDK install.")
        
    jdk.install("20", path=jdk_path)

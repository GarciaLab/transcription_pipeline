import jdk
import os
import sys
from pathlib import Path


def install_bioformats():
    """
    Installs the specific version of loci.tools required for the PIMS Bioformats
    reader.
    """
    jdk.install("20")
    
    environment_path = Path(sys.prefix)
    pims_path = environment_path / "lib" / "site-packages" / "pims"
    loci_tools_path = environment_path / "lib" / "site-packages" / "transcription_pipeline" / "loci_tools.jar"
    
    os.replace(loci_tools_path, pims_path / "loci_tools.jar")
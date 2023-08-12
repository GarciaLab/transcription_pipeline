import jdk
import sys
import os
from pathlib import Path
import warnings


def install_bioformats():
    """
    Installs the specific version of loci.tools required for the PIMS Bioformats
    reader.
    """
    environment_path = Path(sys.prefix)
    pims_path = environment_path / "lib" / "site-packages" / "pims"
    loci_tools_path = (
        environment_path
        / "lib"
        / "site-packages"
        / "transcription_pipeline"
        / "loci_tools.jar"
    )

    try:
        os.replace(loci_tools_path, pims_path / "loci_tools.jar")
    except OSError:
        warnings.warn(
            "`loci_tools.jar` not found in package folder, `install_bioformats_jar` may already have been run."
        )

    jdk_path = environment_path / "library" / "lib"
    jdk.install("20", path=jdk_path)
import pims


def install_bioformats():
    """
    Installs the specific version of loci.tools required for the PIMS Bioformats
    reader.
    """
    pims.bioformats.download_jar(version='6.7')
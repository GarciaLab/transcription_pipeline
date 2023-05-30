import pims

def import_dataset(name_folder):
    """
    Imports and cleans up the data files in the name directory (removes
    incompletely scanned frames) and exports the resulting file as a
    TIFF file along with the relevant metadata. All files are exported
    to a "preprocessed_data" folder in the name folder (created if it
    does not already exist).
    
    :param str name_folder: Path to name folder containing data files.
    :return: 
    """
    
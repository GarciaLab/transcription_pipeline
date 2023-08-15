from .preprocessing import import_data
import warnings
import zarr
from skimage.io import imsave, imread
from pathlib import Path
import pickle
import glob


class DataImport:
    """
    Uses the PIMS Bioformats reader to import each channel of a movie as an
    array, extracting the relevant metadata into respective dictionaries as
    per the documentation for :func:`~preprocessing.import_data.import_dataset`.

    :param str name_folder: Path to name folder containing data files.
    :param bool import_previous: If `True`, attemps to import already collated
        data and preprocessed and pickled metadata dicts from `collated_dataset`
        subdirectory instead of using the PIMS Bioformats reader to extract
        from the original microscopy format.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.

    :ivar channels_full_dataset: list of numpy arrays, with each element of the
        list being a collated dataset for a given channel.
    :ivar original_global_metadata: dictionary of global metadata
        for all files and series in a dataset.
    :ivar original_frame_metadata: dictionary of frame-by-frame metadata
        for all files and series in a dataset.
    :ivar export_global_metadata: list of dictionaries of global
        metadata for the collated dataset, with each element of the list
        corresponding to a channel.
    :ivar export_frame_metadata: list of dictionaries of frame-by-frame
        metadata for the collated dataset, with each element of the list
        corresponding to a channel.
    """

    def __init__(
        self,
        *,
        name_folder,
        import_previous=False,
        trim_series=False,
    ):
        """
        Constructor method. Instantiates class with imported data as attributes as
        per the notation in :func:`~preprocessing.import_data.import_dataset`.
        Class is instantiated as empty if `name_folder=None`.
        """
        if import_previous:
            self.read()
        else:
            self.name_folder = name_folder
            self.trim_series = trim_series
            (
                self.channels_full_dataset,
                self.original_global_metadata,
                self.original_frame_metadata,
                self.export_global_metadata,
                self.export_frame_metadata,
            ) = import_data.import_dataset(self.name_folder, self.trim_series)

    def read(self):
        """
        Imports preprocessed and collated data along with metadata dictionaries extracted
        from a movie and saved to disk in the `name_folder` directory, loading it into
        respective class attributes.
        """
        self.name_folder = name_folder
        name_path = Path(self.name_folder)
        collated_path = name_path / "collated_dataset"

        # Read original metadata dicts (single dictionary for global and single dictionary
        # for frame-by-frame).
        global_metadata_path = collated_path / "original_global_metadata.pkl"
        with open(global_metadata_path, "rb") as f:
            self.original_global_metadata = pickle.load(f)

        frame_metadata_path = collated_path / "original_frame_metadata.pkl"
        with open(frame_metadata_path, "rb") as f:
            self.original_frame_metadata = pickle.load(f)

        # Iterate over files with matching name patterns to find all channels
        global_metadata_file_list = glob.glob(
            str(collated_path / "collated_global_metadata_ch*.pkl")
        )
        global_metadata_file_list.sort()
        self.export_global_metadata = []
        for file in global_metadata_file_list:
            with open(file, "rb") as f:
                channel_global_metadata = pickle.load(f)
            self.export_global_metadata.append(channel_global_metadata)

        frame_metadata_file_list = glob.glob(
            str(collated_path / "collated_frame_metadata_ch*.pkl")
        )
        frame_metadata_file_list.sort()
        self.export_frame_metadata = []
        for file in frame_metadata_file_list:
            with open(file, "rb") as f:
                frame_global_metadata = pickle.load(f)
            self.export_frame_metadata.append(frame_global_metadata)

        # Iterate over data files, and determine format before import
        tiff_file_list = glob.glob(str(collated_path / "*.tiff"))
        zarr_file_list = glob.glob(str(collated_path / "*.zarr"))

        if len(zarr_file_list) > 0:
            file_list = zarr_file_list
            mode = "zarr"
            if len(tiff_file_list) > 0:
                warnings.warn(
                    "".join(
                        [
                            "Multiple usable formats found in `collated_dataset` folder",
                            " defaulting to `zarr`.",
                        ]
                    )
                )
        else:
            file_list = tiff_file_list
            mode = "tiff"

        if len(file_list) == 0:
            raise Exception("No usable files found in `collated_dataset` subdirectory.")

        file_list.sort()

        self.channels_full_dataset = []
        for file in file_list:
            if mode == "zarr":
                self.channels_full_dataset.append(zarr.load(file))
            elif mode == "tiff":
                self.channels_full_dataset.append(imread(file, plugin="tifffile"))

    def save(self, *, mode="zarr"):
        """
        Saves preprocessed and collated channel data in the `name_folder` directory
        in the format requested, and saves the metadata dictionaries in the same
        folder using `pickle`.

        :param mode: Format to save the data arrays.
        :type mode: {'tiff', 'zarr'}
        """
        # Make collated_dataset directory if it doesn't exist
        name_path = Path(self.name_folder)
        collated_path = name_path / "collated_dataset"
        collated_path.mkdir(exist_ok=True)

        global_metadata_path = collated_path / "original_global_metadata.pkl"
        with open(global_metadata_path, "wb") as f:
            pickle.dump(self.original_global_metadata, f)

        frame_metadata_path = collated_path / "original_frame_metadata.pkl"
        with open(frame_metadata_path, "wb") as f:
            pickle.dump(self.original_frame_metadata, f)

        for i, channel_data in enumerate(self.channels_full_dataset):
            # Save metadata to file
            collated_global_path = (
                collated_path / "collated_global_metadata_ch{:02d}.pkl".format(i)
            )
            with open(collated_global_path, "wb") as f:
                pickle.dump(self.export_global_metadata[i], f)

            collated_frame_path = (
                collated_path / "collated_frame_metadata_ch{:02d}.pkl".format(i)
            )
            with open(collated_frame_path, "wb") as f:
                pickle.dump(self.export_frame_metadata[i], f)

            if mode == "tiff":
                # Save data to file for each channel
                filename = "".join(
                    [(self.export_global_metadata[i])["ImageName"], ".tiff"]
                )
                collated_data_path = collated_path / filename
                imsave(collated_data_path, channel_data, plugin="tifffile")

            elif mode == "zarr":
                # Save data to file for each channel
                filename = "".join(
                    [(self.export_global_metadata[i])["ImageName"], ".zarr"]
                )
                collated_data_path = collated_path / filename

                # Convert to zarr
                store = zarr.storage.DirectoryStore(collated_data_path)
                zarr.creation.array(channel_data, store=store)
                store.close()

            else:
                raise Exception("Save mode not recognized.")


class FullEmbryoImport:
    """
    Uses the PIMS Bioformats reader to import each of the images in the
    FullEmbryo directory as a list of arrays with each element corresponding to a channel,
    extracting the relevant metadata into respective dictionaries as per the documentation
    for :func:`~preprocessing.import_data.import_full_embryo`.

    :param str name_folder: Path to name folder containing FullEmbryo directory.
    :param bool import_previous: If `True`, attemps to import already collated
        data and preprocessed and pickled metadata dicts from `preprocessed_full_embryo`
        subdirectory instead of using the PIMS Bioformats reader to extract
        from the original microscopy format.

    :ivar channels_dataset_mid: list of numpy arrays, with each element of the
        list being a channel of the `Mid` image.
    :ivar original_global_metadata_mid: dictionary of global metadata
           for the `Mid` image.
    :ivar original_frame_metadata_mid:  dictionary of frame-by-frame metadata
           for the `Mid` image.
    :ivar export_global_metadata_mid: list of dictionaries of global
        metadata for the `Mid` image, with each element of the list
        corresponding to a channel.
    :ivar export_frame_metadata_mid: list of dictionaries of frame-by-frame
        metadata for the `Mid` image, with each element of the list
        corresponding to a channel. FullEmbryo images are usually single time-points,
        so this is redundant but kept to maintain consistency with the interfact for
        other import classes.
    :ivar channels_dataset_surf: list of numpy arrays, with each element of the
        list being a channel of the `Surf` image.
    :ivar original_global_metadata_surf: dictionary of global metadata
           for the `Surf` image.
    :ivar original_frame_metadata_surf:  dictionary of frame-by-frame metadata
           for the `Surf` image.
    :ivar export_global_metadata_surf: list of dictionaries of global
        metadata for the `Surf` image, with each element of the list
        corresponding to a channel.
    :ivar export_frame_metadata_surf: list of dictionaries of frame-by-frame
        metadata for the `Surf` image, with each element of the list
        corresponding to a channel. FullEmbryo images are usually single time-points,
        so this is redundant but kept to maintain consistency with the interfact for
        other import classes.
    """

    def __init__(
        self,
        *,
        name_folder,
        import_previous=False,
    ):
        """
        Constructor method. Instantiates class with imported data as attributes as
        per the notation in :func:`~preprocessing.import_data.import_full_embryo`.
        Class is instantiated as empty if `name_folder=None`.
        """
        if import_previous:
            self.read()
        else:
            self.name_folder = name_folder

            (
                self.channels_full_dataset_mid,
                self.original_global_metadata_mid,
                self.original_frame_metadata_mid,
                self.export_global_metadata_mid,
                self.export_frame_metadata_mid,
            ) = import_data.import_full_embryo(self.name_folder, "Mid*")

            (
                self.channels_full_dataset_surf,
                self.original_global_metadata_surf,
                self.original_frame_metadata_surf,
                self.export_global_metadata_surf,
                self.export_frame_metadata_surf,
            ) = import_data.import_full_embryo(self.name_folder, "Surf*")

    def read(self):
        """
        Imports preprocessed and collated FullEmbryo data along with metadata dictionaries
        extracted and saved to disk in the `name_folder` directory, loading it into
        respective class attributes.
        """
        self.name_folder = name_folder
        name_path = Path(self.name_folder)
        collated_path = name_path / "preprocessed_full_embryo"

        def _read_saved_fullembryo(name):
            # Read original metadata dicts (single dictionary for global and single dictionary
            # for frame-by-frame).
            global_metadata_path = collated_path / "".join(
                ["original_global_metadata_", name, ".pkl"]
            )
            with open(global_metadata_path, "rb") as f:
                setattr(
                    self, "".join(["original_global_metadata_", name]), pickle.load(f)
                )

            frame_metadata_path = collated_path / "".join(
                ["original_frame_metadata_", name, ".pkl"]
            )
            with open(frame_metadata_path, "rb") as f:
                setattr(
                    self, "".join(["original_frame_metadata_", name]), pickle.load(f)
                )

            # Iterate over files with matching name patterns to find all channels
            global_metadata_file_list = glob.glob(
                str(
                    collated_path
                    / "".join(["collated_global_metadata_", name, "_ch*.pkl"])
                )
            )
            global_metadata_file_list.sort()
            setattr(self, "".join(["export_global_metadata_", name]), [])
            for file in global_metadata_file_list:
                with open(file, "rb") as f:
                    channel_global_metadata = pickle.load(f)
                getattr(self, "".join(["export_global_metadata_", name])).append(
                    channel_global_metadata
                )

            frame_metadata_file_list = glob.glob(
                str(
                    collated_path
                    / "".join(["collated_frame_metadata_", name, "ch*.pkl"])
                )
            )
            frame_metadata_file_list.sort()
            setattr(self, "".join(["export_frame_metadata_", name]), [])
            for file in frame_metadata_file_list:
                with open(file, "rb") as f:
                    frame_global_metadata = pickle.load(f)
                getattr(self, "".join(["export_frame_metadata_", name])).append(
                    frame_global_metadata
                )

            # Iterate over data files, and determine format before import
            tiff_file_list = glob.glob(str(collated_path / "*.tiff"))
            zarr_file_list = glob.glob(str(collated_path / "*.zarr"))

            if len(zarr_file_list) > 0:
                file_list = zarr_file_list
                mode = "zarr"
                if len(tiff_file_list) > 0:
                    warnings.warn(
                        "".join(
                            [
                                "Multiple usable formats found in `collated_dataset` folder",
                                " defaulting to `zarr`.",
                            ]
                        )
                    )
            else:
                file_list = tiff_file_list
                mode = "tiff"

            if len(file_list) == 0:
                raise Exception(
                    "No usable files found in `collated_dataset` subdirectory."
                )

            file_list.sort()

            setattr(self, "".join(["channels_full_dataset_", name]), [])
            for file in file_list:
                if mode == "zarr":
                    getattr(self, "".join(["channels_full_dataset_", name])).append(
                        zarr.load(file)
                    )
                elif mode == "tiff":
                    getattr(self, "".join(["channels_full_dataset_", name])).append(
                        imread(file, plugin="tifffile")
                    )

        _read_saved_fullembryo("mid")
        _read_saved_fullembryo("surf")

    def save(self, *, mode="zarr"):
        """
        Saves preprocessed and collated FullEmbryo data in the `name_folder` directory
        in the format requested, and saves the metadata dictionaries in the same
        folder using `pickle`.

        :param mode: Format to save the data arrays.
        :type mode: {'tiff', 'zarr'}
        """
        # Make collated_dataset directory if it doesn't exist
        name_path = Path(self.name_folder)
        collated_path = name_path / "preprocessed_full_embryo"
        collated_path.mkdir(exist_ok=True)

        def _save_fullembryo(name):
            global_metadata_path = collated_path / "".join(
                ["original_global_metadata_", name, ".pkl"]
            )
            with open(global_metadata_path, "wb") as f:
                original_global_metadata = getattr(
                    self, "".join(["original_global_metadata_", name])
                )
                pickle.dump(original_global_metadata, f)

            frame_metadata_path = collated_path / "".join(
                ["original_frame_metadata_", name, ".pkl"]
            )
            with open(frame_metadata_path, "wb") as f:
                original_frame_metadata = getattr(
                    self, "".join(["original_frame_metadata_", name])
                )
                pickle.dump(original_frame_metadata, f)

            for i, channel_data in enumerate(
                getattr(self, "".join(["channels_full_dataset_", name]))
            ):
                # Save metadata to file
                collated_global_path = collated_path / "".join(
                    ["collated_global_metadata_", name, "_ch{:02d}.pkl".format(i)]
                )
                with open(collated_global_path, "wb") as f:
                    pickle.dump(
                        getattr(self, "".join(["export_global_metadata_", name]))[i], f
                    )

                collated_frame_path = collated_path / "".join(
                    ["collated_frame_metadata_", name, "_ch{:02d}.pkl".format(i)]
                )
                with open(collated_global_path, "wb") as f:
                    pickle.dump(
                        getattr(self, "".join(["export_frame_metadata_", name]))[i], f
                    )

                if mode == "tiff":
                    # Save data to file for each channel
                    filename = "".join(
                        [
                            (
                                getattr(
                                    self, "".join(["export_global_metadata_", name])
                                )[i]
                            )["ImageName"],
                            ".tiff",
                        ]
                    )
                    collated_data_path = collated_path / filename
                    imsave(collated_data_path, channel_data, plugin="tifffile")

                elif mode == "zarr":
                    # Save data to file for each channel
                    filename = "".join(
                        [
                            (
                                getattr(
                                    self, "".join(["export_global_metadata_", name])
                                )[i]
                            )["ImageName"],
                            ".zarr",
                        ]
                    )
                    collated_data_path = collated_path / filename

                    # Convert to zarr
                    zarr.save_array(collated_data_path, channel_data)

                else:
                    raise Exception("Save mode not recognized.")

        _save_fullembryo("mid")
        _save_fullembryo("surf")
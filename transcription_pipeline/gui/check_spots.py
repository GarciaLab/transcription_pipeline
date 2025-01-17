"""
This module provides an interactive graphical interface for spot intensity analysis using Napari. The `CheckSpots` class
allows visualization of spot intensity traces along with error bars and links this data with image and label layers in
Napari. The `CheckSpotsGUI` class serves as an initializer for the graphical interface. The key bindings are as follows:
navigate traces with `a`(previous) and `d` (next). Use `s` to approve a trace and `w` to reject it, with updates
reflected in the plot and dataset. Selecting a label in the Napari viewer syncs with the corresponding trace,
while a red vertical line highlights the currently selected frame in the plot.

Constants:
    :const str DEFAULT_QUANTIFICATION_COLUMN: Default column for quantification, "photometry_flux".
    :const str DEFAULT_ERROR_COLUMN: Default column for error values, "photometry_flux_error".

Functions:
    extract_dataset(dataset_name: str, spot_image_index: int) -> tuple:
        Extracts the dataset and spot label data required for visualization.

    update_plot(ax,ax_frame, curr_pos, frame_min, frame_max, filtered_data, column, error_column, labels_layer=None):
        Updates the plot with trace data for a given particle.

Classes:
    CheckSpots(QWidget):
        Main widget for the Napari spot-checking graphical interface.

    CheckSpotsGUI:
        Control class to initialize the Napari viewer and the `CheckSpots` widget.
"""

from asyncio import current_task

import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QWidget, QVBoxLayout
from napari_matplotlib.base import NapariMPLWidget
import numpy as np
from transcription_pipeline import preprocessing_pipeline, spot_pipeline
from magicgui import magicgui

# Define constants to avoid magic strings
DEFAULT_QUANTIFICATION_COLUMN = "photometry_flux"
DEFAULT_ERROR_COLUMN = "photometry_flux_error"
PLOT_PADDING_FRACTION = 0.05


def extract_dataset(dataset_name, spot_image_index=0):
    """
    Extract the dataset and spot label data required for visualization.

    :param str dataset_name: Name of the dataset to extract.
    :param int spot_image_index: Index of the spot image in channel data. Default is 0.
    :return: A tuple containing spot image data and label data.
    :rtype: tuple
    """
    dataset = preprocessing_pipeline.DataImport(
        name_folder=dataset_name, import_previous=True
    )
    spot_tracker = spot_pipeline.Spot()
    spot_tracker.read_results(
        name_folder=dataset_name, import_all=False, import_params=False
    )
    spot_image_data = dataset.channels_full_dataset[spot_image_index]
    label_data = spot_tracker.reordered_spot_labels
    return spot_image_data, label_data


def update_plot(
    ax,
    curr_pos,
    frame_min,
    frame_max,
    filtered_data,
    column=DEFAULT_QUANTIFICATION_COLUMN,
    error_column=DEFAULT_ERROR_COLUMN,
    labels_layer=None,
):
    """
    Update the plot axis with trace data for the current particle.

    :param ax: Matplotlib axis to draw the plot.
    :param int curr_pos: The current position in the dataset.
    :param int frame_min: Earliest frame in current trace.
    :param int frame_max: Latest frame in current trace.
    :param pandas.DataFrame filtered_data: The data containing trace and associated information.
    :param str column: Name of the column for quantification data. Default is "photometry_flux".
    :param str error_column: Name of the column for error data. Default is "photometry_flux_error".
    :param Labels labels_layer: Napari labels layer for associating selected labels. Default is None.
    """
    frames = filtered_data.loc[curr_pos, "frame"]
    trace = filtered_data.loc[curr_pos, column]
    trace_err = filtered_data.loc[curr_pos, error_column]
    ax.cla()
    ax.set_title(f"Particle {filtered_data.loc[curr_pos, 'particle']}", color="white")
    ax.errorbar(frames, trace, yerr=trace_err, fmt=".", elinewidth=0.5, markersize=3)
    ax.set_ylabel("Spot intensity (AU)", color="white")
    ax.set_xlabel("Frame", color="white")
    padding = PLOT_PADDING_FRACTION * (frame_max - frame_min)
    ax.set_xlim(frame_min - padding, frame_max + padding)
    if labels_layer:
        labels_layer.selected_label = filtered_data.loc[curr_pos, "particle"]


def update_approval(
    ax_box,
    curr_pos,
    filtered_data,
):
    """
    Update the indicator box with the approval state (selected/rejected) of the trace.

    :param ax_box: Matplotlib axis to draw the indicator box.
    :param int curr_pos: The current position in the dataset.
    :param pandas.DataFrame filtered_data: The data containing trace and associated information.
    """
    # these are matplotlib.patch.Patch properties
    ax_box.cla()
    approval_state = filtered_data.loc[curr_pos, "Selected"]
    if approval_state:
        alpha_withdrawn = 0
        alpha_selected = 0.5
    else:
        alpha_withdrawn = 0.5
        alpha_selected = 0
    props_withdrawn = dict(
        boxstyle="square", facecolor="magenta", alpha=alpha_withdrawn
    )
    props_selected = dict(boxstyle="square", facecolor="green", alpha=alpha_selected)

    # Add text boxes to keep track of approval state

    ax_box.text(
        0.84,
        1.02,
        "[S]elected",
        transform=ax_box.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props_selected,
        color="white",
    )
    ax_box.text(
        1,
        1.02,
        "[W]ithdrawn",
        transform=ax_box.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props_withdrawn,
        color="white",
    )


def update_frame_marker(
    ax_frame,
    frame_min,
    frame_max,
    napari_viewer,
):
    """
    Update the indicator box with the approval state (selected/rejected) of the trace.

    :param ax_frame: Matplotlib axis to draw the vertical frame marker.
    :param frame_min: Earliest frame in current trace.
    :param frame_max: Latest frame in current trace.
    :param napari.viewer.Viewer napari_viewer: The Napari viewer to which this widget is connected.
    """
    # Add frame marker
    ax_frame.cla()
    current_frame = napari_viewer.dims.point[0]
    if frame_min <= current_frame <= frame_max:
        ax_frame.axvline(x=current_frame, color="red", linestyle="--")


class CheckSpots(QWidget):
    """
    Main widget for visualizing and analyzing spot intensity data using Napari.

    :param napari.viewer.Viewer napari_viewer: The Napari viewer to which this widget is connected.
    :param array spot_image_data: Image data for the spots. Default is None.
    :param array label_data: Label data corresponding to the spots. Default is None.
    :param str dataset_name: Name of the dataset being analyzed. Default is None.
    :param int spot_image_index: Index for accessing the relevant spot image channel. Default is 0.
    :param pandas.DataFrame compiled_data: Compiled DataFrame containing spot data. Default is None.
    :param str quantification_column: Column name for quantification data. Default is `DEFAULT_QUANTIFICATION_COLUMN`.
    :param str error_column: Column name for error data. Default is `DEFAULT_ERROR_COLUMN`.
    :param QWidget parent: Parent widget. Default is None.
    """

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer,
        spot_image_data=None,
        label_data=None,
        dataset_name=None,
        spot_image_index=0,
        compiled_data=None,
        quantification_column=DEFAULT_QUANTIFICATION_COLUMN,
        error_column=DEFAULT_ERROR_COLUMN,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.dataset_name = dataset_name
        self.quantification_column = quantification_column
        self.error_column = error_column
        if spot_image_data is None and label_data is None:
            self.spot_image_data, self.label_data = extract_dataset(
                dataset_name, spot_image_index
            )
        else:
            self.spot_image_data, self.label_data = spot_image_data, label_data
        self.napari_viewer = napari_viewer
        self.curr_pos = 0
        self.curr_spot_coord = None

        # Initialize approval column for manual curation. Needs to be initialized before UI.
        self.compiled_data = compiled_data
        if "Selected" not in self.compiled_data.columns:
            self.compiled_data["Selected"] = True
        self.trace_frame_min = self.compiled_data.loc[self.curr_pos, "frame"][0]
        self.trace_frame_max = self.compiled_data.loc[self.curr_pos, "frame"][-1]

        self._initialize_ui()

    def _initialize_ui(self):
        """
        Private method to initialize the UI elements, including image and labels layers,
        Matplotlib widgets, and event connections.
        """
        self.setLayout(QVBoxLayout())
        self.spots_layer = self.napari_viewer.add_image(
            self.spot_image_data, name="Spots"
        )
        self.labels_layer = self.napari_viewer.add_labels(
            self.label_data, name="Labels"
        )
        plot_widget = NapariMPLWidget(self.napari_viewer, parent=self)
        self.layout().addWidget(plot_widget)
        self.figure = plot_widget.figure
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.tick_params(colors="white")

        self.ax_frame = self.ax.twinx()
        self.ax_frame.get_yaxis().set_visible(False)

        self.ax_box = self.ax.twiny()
        self.ax_box.get_xaxis().set_visible(False)
        self.ax_box.get_yaxis().set_visible(False)

        # noinspection PyTypeChecker
        self.update_all_plots()
        self.figure.canvas.draw()
        self.num_traces = len(self.compiled_data.index)

        @self.napari_viewer.bind_key("a", overwrite=True)
        def plot_previous_trace(_viewer):
            """
            Bind the "a" key to plot the previous trace.
            """
            if self.curr_pos > 0:
                self.curr_pos -= 1
                # noinspection PyTypeChecker
                self.update_trace_bounds()
                self.update_all_plots()
                self.figure.canvas.draw()
            if "centroid" in self.compiled_data.columns:
                self.curr_spot_coord = self.compiled_data.loc[self.curr_pos, "centroid"]

        @self.napari_viewer.bind_key("d", overwrite=True)
        def plot_next_trace(_viewer):
            """
            Bind the "d" key to plot the next trace.
            """
            if self.curr_pos < self.num_traces - 1:
                self.curr_pos += 1
                # noinspection PyTypeChecker
                self.update_trace_bounds()
                self.update_all_plots()
                self.figure.canvas.draw()
            if "centroid" in self.compiled_data.columns:
                self.curr_spot_coord = self.compiled_data.loc[self.curr_pos, "centroid"]

        @self.napari_viewer.bind_key("s", overwrite=True)
        def select_trace(_viewer):
            """
            Bind the "s" key to select (approve) the trace.
            """
            self.compiled_data.loc[self.curr_pos, "Selected"] = True
            # noinspection PyTypeChecker
            update_approval(
                self.ax_box,
                self.curr_pos,
                self.compiled_data,
            )
            self.figure.canvas.draw()

        @self.napari_viewer.bind_key("w", overwrite=True)
        def withdraw_trace(_viewer):
            """
            Bind the "w" key to withdraw (reject) the trace.
            """
            self.compiled_data.loc[self.curr_pos, "Selected"] = False
            # noinspection PyTypeChecker
            update_approval(
                self.ax_box,
                self.curr_pos,
                self.compiled_data,
            )
            self.figure.canvas.draw()

        @self.napari_viewer.dims.events.point.connect
        def mark_frame(_viewer):
            """
            Highlight the current frame based on the viewer's point dimension.
            """
            # noinspection PyTypeChecker
            update_frame_marker(
                self.ax_frame,
                self.trace_frame_min,
                self.trace_frame_max,
                self.napari_viewer,
            )
            self.figure.canvas.draw()

        @self.labels_layer.events.selected_label.connect
        def label_selection_callback(_event: Event):
            """
            Update the plot based on the selected label from the labels layer.
            """
            selected_label = self.labels_layer.selected_label
            label_indices = np.argwhere(
                selected_label == self.compiled_data["particle"]
            )
            if label_indices.size > 0:
                self.curr_pos = int(label_indices[0][0])
                self.update_trace_bounds()
                self.update_all_plots()
                self.figure.canvas.draw()
            if "centroid" in self.compiled_data.columns:
                self.curr_spot_coord = self.compiled_data.loc[self.curr_pos, "centroid"]

    def update_all_plots(self):
        # noinspection PyTypeChecker
        update_plot(
            self.ax,
            self.curr_pos,
            self.trace_frame_min,
            self.trace_frame_max,
            self.compiled_data,
            self.quantification_column,
            self.error_column,
            self.labels_layer,
        )
        update_approval(
            self.ax_box,
            self.curr_pos,
            self.compiled_data,
        )
        # noinspection PyTypeChecker
        update_frame_marker(
            self.ax_frame,
            self.trace_frame_min,
            self.trace_frame_max,
            self.napari_viewer,
        )

    def update_trace_bounds(self):
        self.trace_frame_min = self.compiled_data.loc[self.curr_pos, "frame"][0]
        self.trace_frame_max = self.compiled_data.loc[self.curr_pos, "frame"][-1]


class CheckSpotsGUI:
    """
    Initializes the Napari viewer and adds the `CheckSpots` graphical interface as a dock widget.

    :param array spot_channel: Image data for the spots. Default is None.
    :param array labels: Label data corresponding to the spots. Default is None.
    :param str dataset_name: Name of the dataset being analyzed. Default is None.
    :param int spot_channel_index: Index for accessing the relevant spot image channel. Default is 0.
    :param pandas.DataFrame compiled_dataframe: Compiled DataFrame containing spot data. Default is None.
    :param str quantification_column: Column name for quantification data. Default is `DEFAULT_QUANTIFICATION_COLUMN`.
    :param str error_column: Column name for error data. Default is `DEFAULT_ERROR_COLUMN`.
    :param QWidget parent: Parent widget. Default is None.
    """

    def __init__(
        self,
        spot_channel=None,
        labels=None,
        dataset_name=None,
        spot_channel_index=0,
        compiled_dataframe=None,
        quantification_column=DEFAULT_QUANTIFICATION_COLUMN,
        error_column=DEFAULT_ERROR_COLUMN,
        parent=None,
    ):
        # Initialize the napari viewer
        viewer = napari.Viewer()

        # Initialize the CheckSpots widget
        self.CheckSpots = CheckSpots(
            napari_viewer=viewer,
            spot_image_data=spot_channel,
            label_data=labels,
            dataset_name=dataset_name,
            spot_image_index=spot_channel_index,
            compiled_data=compiled_dataframe,
            quantification_column=quantification_column,
            error_column=error_column,
            parent=parent,
        )

        # Create a unified widget to combine CheckSpots and the magicgui widget
        unified_widget = QWidget(parent=parent)
        unified_layout = QVBoxLayout()
        unified_widget.setLayout(unified_layout)

        # Add the CheckSpots widget to the unified container
        unified_layout.addWidget(self.CheckSpots)

        # Define the "Fix spot coordinates" checkbox widget
        self.fix_spot_coordinates = False

        @magicgui(auto_call=True, spot_coordinate={"label": "Fix spot coordinates"})
        def coordinate_change_widget(spot_coordinate: bool):
            self.fix_spot_coordinates = spot_coordinate

        # Add the coordinate change widget to the unified container
        unified_layout.addWidget(coordinate_change_widget.native)

        # Set up the recenter camera functionality
        @viewer.dims.events.point.connect
        def recenter_camera(_viewer):
            # Check if the frame exists in this trace
            trace_frames = self.CheckSpots.compiled_data.loc[
                self.CheckSpots.curr_pos, "frame"
            ]
            if (viewer.dims.point[0] in trace_frames) & self.fix_spot_coordinates:
                viewer.camera.center = self.CheckSpots.curr_spot_coord[
                    int(np.argwhere(viewer.dims.point[0] == trace_frames)[0][0])
                ]

        # Finally, add the unified widget to the viewer's dock
        viewer.window.add_dock_widget(unified_widget, area="right", name="CheckSpots")

        # Run the napari event loop
        napari.run()

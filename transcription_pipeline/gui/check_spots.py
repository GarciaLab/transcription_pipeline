import napari
from napari.utils.events import Event
from qtpy.QtWidgets import QWidget, QVBoxLayout
from napari_matplotlib.base import NapariMPLWidget
import numpy as np

# noinspection PyUnresolvedReferences
from transcription_pipeline import preprocessing_pipeline

# noinspection PyUnresolvedReferences
from transcription_pipeline import spot_pipeline


def extract_dataset(
    dataset_name,
    spot_channel=0,
):
    # Extract respective class attributes
    dataset = preprocessing_pipeline.DataImport(
        name_folder=dataset_name,
        import_previous=True,
    )

    spot_tracking = spot_pipeline.Spot()

    spot_tracking.read_results(
        name_folder=dataset_name,
        import_all=False,
        import_params=False,
    )
    movie = dataset.channels_full_dataset[spot_channel]
    labels = spot_tracking.reordered_spot_labels

    return movie, labels


def plot_trace(
    ax,
    curr_pos,
    filtered_compiled_traces,
    quantification_column="photometry_flux",
    quantification_error_column="photometry_flux_error",
    labels_layer=None,
):
    frames = filtered_compiled_traces.loc[curr_pos, "frame"]
    trace = filtered_compiled_traces.loc[curr_pos, quantification_column]

    trace_err = filtered_compiled_traces.loc[curr_pos, quantification_error_column]

    ax.cla()

    title = "particle"
    ax.set_title(
        f"Particle {filtered_compiled_traces.loc[curr_pos, title]}", color="white"
    )

    ax.errorbar(
        frames,
        trace,
        yerr=trace_err,
        fmt=".",
        elinewidth=0.5,
        markersize=3,
    )

    ax.set_ylabel("Spot intensity (AU)", color="white")
    ax.set_xlabel("Frame", color="white")

    # Select particles in labels layer
    if labels_layer is not None:
        labels_layer.selected_label = filtered_compiled_traces.loc[curr_pos, "particle"]


class CheckSpots(QWidget):
    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer,
        spot_channel=None,
        labels=None,
        dataset_name=None,
        spot_channel_index=0,
        compiled_dataframe=None,
        quantification_column="photometry_flux",
        quantification_error_column="photometry_flux_error",
        parent=None,
    ):
        super().__init__(parent=parent)
        self.dataset_name = dataset_name
        self.quantification_column = quantification_column
        self.quantification_error_column = quantification_error_column

        # Extract dataset
        if (spot_channel is None) & (labels is None):
            self.movie, self.labels = extract_dataset(dataset_name, spot_channel_index)
        else:
            self.movie, self.labels = spot_channel, labels
        self.compiled_dataframe = compiled_dataframe

        # Set up the layout for the widget
        self.setLayout(QVBoxLayout())

        # Set up viewer
        self.napari_viewer = napari_viewer
        self.spots_layer = napari_viewer.add_image(self.movie, name="Spots")
        self.labels_layer = napari_viewer.add_labels(self.labels, name="Labels")

        # Set up the Matplotlib plot widget
        plot_widget = NapariMPLWidget(napari_viewer, parent=self)
        self.layout().addWidget(plot_widget)

        # Access the figure and axes of the plot widget
        self.figure = plot_widget.figure
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.tick_params(colors="white")

        # Initialize trace plot
        self.curr_pos = 0
        plot_trace(
            self.ax,
            self.curr_pos,
            self.compiled_dataframe,
            self.quantification_column,
            self.quantification_error_column,
            self.labels_layer,
        )
        self.figure.canvas.draw()

        num_traces = len(self.compiled_dataframe.index)

        # Example: Connect napari events
        @napari_viewer.bind_key("a", overwrite=True)
        def plot_previous_trace(napari_viewer):
            print("test")
            if self.curr_pos > 0:
                self.curr_pos -= 1
            self.ax.cla()
            plot_trace(
                self.ax,
                self.curr_pos,
                self.compiled_dataframe,
                self.quantification_column,
                self.quantification_error_column,
                self.labels_layer,
            )
            self.figure.canvas.draw()

        @napari_viewer.bind_key("d", overwrite=True)
        def plot_next_trace(napari_viewer):
            if self.curr_pos < num_traces - 1:
                self.curr_pos += 1
            self.ax.cla()
            plot_trace(
                self.ax,
                self.curr_pos,
                self.compiled_dataframe,
                self.quantification_column,
                self.quantification_error_column,
                self.labels_layer,
            )
            self.figure.canvas.draw()

        @self.labels_layer.events.selected_label.connect
        def label_selection_callback(event: Event):
            label = self.labels_layer.selected_label
            self.curr_pos = np.argwhere(label == self.compiled_dataframe["particle"])[
                0
            ][0]

            self.ax.cla()
            plot_trace(
                self.ax,
                self.curr_pos,
                self.compiled_dataframe,
                self.quantification_column,
                self.quantification_error_column,
                self.labels_layer,
            )
            self.figure.canvas.draw()


# Main function GUI
class CheckSpotsGUI:
    def __init__(
        self,
        *,
        spot_channel=None,
        labels=None,
        dataset_name=None,
        spot_channel_index=0,
        compiled_dataframe=None,
        quantification_column="photometry_flux",
        quantification_error_column="photometry_flux_error",
        parent=None,
    ):
        viewer = napari.Viewer()

        # Add the custom widget to the Napari viewer as a dock widget
        gui_plugin = CheckSpots(
            viewer,
            spot_channel=spot_channel,
            labels=labels,
            dataset_name=dataset_name,
            spot_channel_index=spot_channel_index,
            compiled_dataframe=compiled_dataframe,
            quantification_column=quantification_column,
            quantification_error_column=quantification_error_column,
            parent=parent,
        )
        viewer.window.add_dock_widget(gui_plugin, area="right", name="CheckSpots")

        # Start the Napari application
        napari.run()

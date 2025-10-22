"""
make_movie.py

This script generates Napari animations based on a configuration YAML file.

How to Use:
1. Create a corresponding YAML file for each experiment, e.g., `experiment_config.yaml`.
   - The YAML file should include parameters like paths, layer settings, and camera settings.
2. Run the script from the terminal with the following command:
   ```bash
   python make_movie.py --config experiment_config.yaml
   ```
3. The script will process the settings and create an animation.
   The output video will be saved based on the `output` path specified in the YAML file.

Dependencies:
- Python 3.9+
- napari
- napari-animation
- pyyaml
"""

import yaml
import napari
from napari_animation import Animation


def main(config_path):
    """
    Main function to generate a Napari animation based on the provided YAML configuration.

    Args:
        config_path (str): Path to the YAML configuration file.
    """

    #TODO: use utils/check_pyramids.py to check if the zarrs have pyramids.

    # Load configuration from YAML
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load paths and parameters from configuration
    paths = config["paths"]
    viewer_params = config["viewer_params"]
    camera_settings = config["camera"]

    # Get start and end frames (if not provided, default to full range)
    start_frame = viewer_params.get("start_frame", 0)  # Default to the first frame
    end_frame = viewer_params.get("end_frame", None)  # Default to the last frame

    # Initialize Napari viewer
    viewer = napari.Viewer()

    # Load datasets and settings
    h2b_layer = viewer.open(paths["h2b"],
                            plugin="napari-ome-zarr",
                            **config["h2b_settings"])
    masked_layer = viewer.open(paths["masked"],
                               plugin="napari-ome-zarr",
                               **config["masked_settings"])

    # Add high-resolution data for each layer
    viewer.add_image(h2b_layer[0].data[0],
                     name="H2B (High-Res)",
                     **config["h2b_settings"])
    viewer.add_image(masked_layer[0].data[0],
                     name="segments_masked (High-Res)",
                     **config["masked_settings"])

    # Remove original layers for clarity
    viewer.layers.remove("H2B")
    if "MCP" in viewer.layers:
        viewer.layers.remove("MCP")
    viewer.layers.remove("segments_masked")

    # Set scale for each layer
    for layer in viewer.layers:
        layer.scale = viewer_params["scale"]

    # Set viewer dimensionality
    viewer.dims.ndisplay = viewer_params["ndisplay"]
    viewer.dims.set_current_step(0, viewer_params["current_step"])

    # Set camera settings
    viewer.camera.angles = camera_settings["angles"]
    viewer.camera.zoom = camera_settings["zoom"]
    viewer.camera.center = tuple(camera_settings["center"])

    # Get total frames from the first layer dataset
    total_frames = viewer.layers[0].data.shape[0]  # Assuming the first axis is time

    # Adjust end_frame to the total number of frames if not provided or out of range
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # Validate start_frame and end_frame
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"Invalid start_frame: {start_frame}. Must be in the range [0, {total_frames - 1}].")
    if end_frame <= start_frame:
        raise ValueError(f"Invalid end_frame: {end_frame}. Must be greater than start_frame.")

    # Create and save animation
    animation = Animation(viewer)
    viewer.dims.set_point(0, start_frame)
    animation.capture_keyframe()
    viewer.dims.set_point(0, end_frame - 1)
    animation.capture_keyframe(steps=end_frame - start_frame - 1)

    # Save the animation to the specified output path
    animation.animate(paths["output"], quality=9, fps=camera_settings["fps"])
    print(f"Animation saved to {paths['output']} (frames {start_frame} to {end_frame})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate videos in Napari using a YAML configuration file.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config)

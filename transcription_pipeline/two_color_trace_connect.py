import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PIPELINE_PATH = "/mnt/Data6/Enze/transcription_pipeline"
DEFAULT_FOLDER_PATH = (
    "/Data/2026-04-05-Enze/"
    "IntB-1DgVw-MS2-LacZ-PP7_MCP-mSG_PCP_Halo_JF552_011"
)
DEFAULT_TRACE_LEN_CUTOFF = 60
DEFAULT_TRACE_CENTER_CUTOFF_PX = 20


def build_trace_dataframe(spot_df: pd.DataFrame, trace_len_cutoff: int) -> pd.DataFrame:
    traces = []

    for label_id in sorted(spot_df["particle"].dropna().unique()):
        spot_trace = spot_df[spot_df["particle"] == label_id].sort_values("t_s")
        trace_time = spot_trace["t_s"]
        trace_signal = spot_trace["intensity_from_neighborhood"]
        trace_signal_std = spot_trace["intensity_std_error_from_neighborhood"]
        trace_background = spot_trace["background_intensity_from_neighborhood"]
        trace_background_std = spot_trace[
            "background_intensity_std_error_from_neighborhood"
        ]

        spot_center = [
            spot_trace["z"].mean(),
            spot_trace["y"].mean(),
            spot_trace["x"].mean(),
        ]

        trace = pd.DataFrame(
            {
                "particle": [label_id],
                "length": [len(trace_time)],
                "spot_center": [spot_center],
                "t_s": [trace_time.to_numpy()],
                "intensity_from_neighborhood": [trace_signal.to_numpy()],
                "intensity_std_error_from_neighborhood": [trace_signal_std.to_numpy()],
                "background_intensity_from_neighborhood": [trace_background.to_numpy()],
                "background_intensity_std_error_from_neighborhood": [
                    trace_background_std.to_numpy()
                ],
            }
        )

        if trace["length"].iloc[0] >= trace_len_cutoff:
            traces.append(trace)

    if not traces:
        return pd.DataFrame(
            columns=[
                "particle",
                "length",
                "spot_center",
                "t_s",
                "intensity_from_neighborhood",
                "intensity_std_error_from_neighborhood",
                "background_intensity_from_neighborhood",
                "background_intensity_std_error_from_neighborhood",
            ]
        )

    return pd.concat(traces, ignore_index=True)


def pair_traces(
    traces_ch00: pd.DataFrame,
    traces_ch01: pd.DataFrame,
    trace_center_cutoff_px: float,
) -> pd.DataFrame:
    candidate_pairs = []

    for _, trace_ch00 in traces_ch00.iterrows():
        center_ch00 = np.asarray(trace_ch00["spot_center"], dtype=float)
        for _, trace_ch01 in traces_ch01.iterrows():
            center_ch01 = np.asarray(trace_ch01["spot_center"], dtype=float)
            center_distance = np.linalg.norm(center_ch00 - center_ch01)
            if center_distance <= trace_center_cutoff_px:
                candidate_pairs.append(
                    {
                        "particle_ch00": trace_ch00["particle"],
                        "particle_ch01": trace_ch01["particle"],
                        "center_distance_px": center_distance,
                    }
                )

    candidate_pairs = sorted(candidate_pairs, key=lambda pair: pair["center_distance_px"])
    paired_particles_ch00 = set()
    paired_particles_ch01 = set()
    traces_2color_rows = []

    for candidate_pair in candidate_pairs:
        particle_ch00 = candidate_pair["particle_ch00"]
        particle_ch01 = candidate_pair["particle_ch01"]

        if particle_ch00 in paired_particles_ch00 or particle_ch01 in paired_particles_ch01:
            continue

        particle_info_ch00 = traces_ch00[
            traces_ch00["particle"] == particle_ch00
        ].reset_index(drop=True)
        particle_info_ch01 = traces_ch01[
            traces_ch01["particle"] == particle_ch01
        ].reset_index(drop=True)

        traces_2color_rows.append(
            {
                "pair_id": len(traces_2color_rows),
                "particle_ids": [particle_ch00, particle_ch01],                
                "center_distance_px": candidate_pair["center_distance_px"],
                "particle_info": [particle_info_ch00, particle_info_ch01],
            }
        )

        paired_particles_ch00.add(particle_ch00)
        paired_particles_ch01.add(particle_ch01)

    return pd.DataFrame(traces_2color_rows)


def connect_two_color_traces(
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    folder_path: str = DEFAULT_FOLDER_PATH,
    trace_len_cutoff: int = DEFAULT_TRACE_LEN_CUTOFF,
    trace_center_cutoff_px: float = DEFAULT_TRACE_CENTER_CUTOFF_PX,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_root = Path(pipeline_path)
    folder_relative = folder_path.lstrip("/")
    dataset_root = pipeline_root / folder_relative

    dataset_path_ch00 = dataset_root / "spot_analysis_results_Ch00" / "spot_dataframe.pkl"
    dataset_path_ch01 = dataset_root / "spot_analysis_results_Ch01" / "spot_dataframe.pkl"

    spot_df_ch00 = pd.read_pickle(dataset_path_ch00)
    spot_df_ch01 = pd.read_pickle(dataset_path_ch01)

    traces_ch00 = build_trace_dataframe(spot_df_ch00, trace_len_cutoff)
    traces_ch01 = build_trace_dataframe(spot_df_ch01, trace_len_cutoff)
    traces_2color = pair_traces(
        traces_ch00=traces_ch00,
        traces_ch01=traces_ch01,
        trace_center_cutoff_px=trace_center_cutoff_px,
    )
    output_path = dataset_root / "spot_2color_dataframe.pkl"
    traces_2color.to_pickle(output_path)

    return traces_ch00, traces_ch01, traces_2color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect two-color fluorescence traces by spot-center proximity."
    )
    parser.add_argument(
        "--pipeline_path",
        default=DEFAULT_PIPELINE_PATH,
        help="Root path of the transcription pipeline repository.",
    )
    parser.add_argument(
        "--folder_path",
        default=DEFAULT_FOLDER_PATH,
        help="Dataset folder path relative to pipeline_path or as an absolute-style repo path.",
    )
    parser.add_argument(
        "--trace_len_cutoff",
        type=int,
        default=DEFAULT_TRACE_LEN_CUTOFF,
        help="Minimum number of timepoints required to keep a trace.",
    )
    parser.add_argument(
        "--trace_center_cutoff_px",
        type=float,
        default=DEFAULT_TRACE_CENTER_CUTOFF_PX,
        help="Maximum allowed spot-center distance in pixels for pairing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces_ch00, traces_ch01, traces_2color = connect_two_color_traces(
        pipeline_path=args.pipeline_path,
        folder_path=args.folder_path,
        trace_len_cutoff=args.trace_len_cutoff,
        trace_center_cutoff_px=args.trace_center_cutoff_px,
    )

    print(f"Channel 00 traces kept: {len(traces_ch00)}")
    print(f"Channel 01 traces kept: {len(traces_ch01)}")
    print(f"Two-color pairs found: {len(traces_2color)}")
    print(f"Saved two-color dataframe to: {args.folder_path}/spot_2color_dataframe.pkl")


if __name__ == "__main__":
    main()

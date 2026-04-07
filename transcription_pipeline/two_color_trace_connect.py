import pandas as pd

pipeline_path ='/mnt/Data6/Enze/transcription_pipeline'

folder_path = "/Data/2026-04-05-Enze/IntB-1DgVw-MS2-LacZ-PP7_MCP-mSG_PCP_Halo_JF552_011"

dataset_path_Ch00 = pipeline_path + folder_path + "/spot_analysis_results_Ch00/spot_dataframe.pkl"
dataset_path_Ch01 = pipeline_path + folder_path + "/spot_analysis_results_Ch01/spot_dataframe.pkl"

spot_df_Ch00 = pd.read_pickle(dataset_path_Ch00)
spot_df_Ch01 = pd.read_pickle(dataset_path_Ch01)

trace_len_cutoff = 60

num_particles_Ch00 = spot_df_Ch00["particle"].max()
traces_Ch00 = pd.DataFrame({})

num_particles_Ch01 = spot_df_Ch01["particle"].max()
traces_Ch01 = pd.DataFrame({})

for label_id in range(num_particles_Ch00):
    spot_trace = spot_df_Ch00[spot_df_Ch00["particle"] == label_id].sort_values("t_s")
    trace_time = spot_trace["t_s"]
    trace_signal = spot_trace["intensity_from_neighborhood"]
    trace_signal_std = spot_trace["intensity_std_error_from_neighborhood"]
    trace_background = spot_trace["background_intensity_from_neighborhood"]
    trace_background_std = spot_trace["background_intensity_std_error_from_neighborhood"]
    z_all = spot_trace["z"]
    y_all = spot_trace["y"]
    x_all = spot_trace["x"]
    spot_center = [z_all.mean(), y_all.mean(), x_all.mean()]
    trace = pd.DataFrame({"particle" : [label_id], 
                          "length" : [len(trace_time)],
                          "spot_center" : [spot_center],
                          "t_s" : [trace_time.to_numpy()], 
                          "intensity_from_neighborhood" : [trace_signal.to_numpy()],
                          "intensity_std_error_from_neighborhood" : [trace_signal_std.to_numpy()],
                          "background_intensity_from_neighborhood" : [trace_background.to_numpy()],
                          "background_intensity_std_error_from_neighborhood" : [trace_background_std.to_numpy()]
                          })
    if trace["length"].iloc[0] >= trace_len_cutoff:
        traces_Ch00 = pd.concat([traces_Ch00,trace], ignore_index=True)

for label_id in range(num_particles_Ch01):
    spot_trace = spot_df_Ch01[spot_df_Ch01["particle"] == label_id].sort_values("t_s")
    trace_time = spot_trace["t_s"]
    trace_signal = spot_trace["intensity_from_neighborhood"]
    trace_signal_std = spot_trace["intensity_std_error_from_neighborhood"]
    trace_background = spot_trace["background_intensity_from_neighborhood"]
    trace_background_std = spot_trace["background_intensity_std_error_from_neighborhood"]
    z_all = spot_trace["z"]
    y_all = spot_trace["y"]
    x_all = spot_trace["x"]
    spot_center = [z_all.mean(), y_all.mean(), x_all.mean()]
    trace = pd.DataFrame({"particle" : [label_id], 
                          "length" : [len(trace_time)],
                          "spot_center" : [spot_center],
                          "t_s" : [trace_time.to_numpy()], 
                          "intensity_from_neighborhood" : [trace_signal.to_numpy()],
                          "intensity_std_error_from_neighborhood" : [trace_signal_std.to_numpy()],
                          "background_intensity_from_neighborhood" : [trace_background.to_numpy()],
                          "background_intensity_std_error_from_neighborhood" : [trace_background_std.to_numpy()]
                          })
    if trace["length"].iloc[0] >= trace_len_cutoff:
        traces_Ch01 = pd.concat([traces_Ch01,trace], ignore_index=True)
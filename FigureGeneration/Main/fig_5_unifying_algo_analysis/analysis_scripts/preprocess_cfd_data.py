import sys

import numpy as np
import pandas

from splitflow.data_paths import data_path



class _Config:
    def configure_gurobi_env(self):
        pass  # Gurobi picked up from the default license location / GRB_LICENSE_FILE


config = _Config()

from braid_analysis import braid_analysis_plots
from braid_analysis import flymath

if __name__ == '__main__':

    # NOTE: this raw CFD simulation output (the un-preprocessed sibling of
    # raw_data/cfd_sim_circling_casting_all_wind_conditions_preprocessed.parquet)
    # is not present in raw_data/ as of this reorganization -- place it there
    # before running this script.
    parquet_filename = str(data_path('sim.casting_constwind_fixed'))
    df_cfd = pandas.read_parquet(parquet_filename)
    df_cfd = df_cfd.reset_index(drop=True)
    update_keys = {'heading': 'course',
                   'ground speed': 'speed_xy'}
    df_cfd = df_cfd.rename(columns=update_keys)
    df_cfd['time_relative_to_flash'] = df_cfd['time stamp']/1000.
    df_cfd['obj_id_unique_event'] = df_cfd['obj_id_unique']
    df_cfd = flymath.assign_course_and_ang_vel_to_dataframe(df_cfd)

    df_cfd.to_parquet(str(data_path('sim.casting_constwind_fixed_preprocessed', must_exist=False)))

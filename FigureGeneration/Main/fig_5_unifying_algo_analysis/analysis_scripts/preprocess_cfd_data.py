import sys
from pathlib import Path

import numpy as np
import pandas


# Data locations. This script lives 4 levels under the repo root:
#   FigureGeneration/Main/fig_5_unifying_algo_analysis/analysis_scripts/<this>.py
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA = _REPO_ROOT / "Data"


class _Config:
    RAW_DATA_DIR = _DATA / "Experimental_Fly_Data"
    PREPROCESSED_DATA_DIR = _DATA / "Unifying_Algo_Results" / "Supplemental"

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
    parquet_filename = str(config.RAW_DATA_DIR / 'casting_sim_constwind_n1000_fixed.parquet')
    df_cfd = pandas.read_parquet(parquet_filename)
    df_cfd = df_cfd.reset_index(drop=True)
    update_keys = {'heading': 'course',
                   'ground speed': 'speed_xy'}
    df_cfd = df_cfd.rename(columns=update_keys)
    df_cfd['time_relative_to_flash'] = df_cfd['time stamp']/1000.
    df_cfd['obj_id_unique_event'] = df_cfd['obj_id_unique']
    df_cfd = flymath.assign_course_and_ang_vel_to_dataframe(df_cfd)

    df_cfd.to_parquet(str(config.RAW_DATA_DIR / 'casting_sim_constwind_n1000_fixed_preprocessed.parquet'))

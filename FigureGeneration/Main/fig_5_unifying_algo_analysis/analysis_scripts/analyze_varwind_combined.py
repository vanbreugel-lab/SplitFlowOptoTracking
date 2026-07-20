import argparse
import sys
from pathlib import Path

import pandas as pd


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
config.configure_gurobi_env()

from align_course_direction_analysis import unifying_algo_analysis as uaa
from align_course_direction_analysis import unifying_algo_plots as uap

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on unifying algo data.')
    parser.add_argument('--include-translation', action='store_true', default=False,
                        help='Include translation in the affine transform (default: False)')
    args = parser.parse_args()

    # Load the data
    # NOTE: originally referenced the no-longer-present
    # 'C1WT_varwind_combined_optotrigger_fancontrol_trimmed.parquet' (no directory
    # prefix); every other notebook/script in this repo reads the file below, so
    # that is almost certainly the intended input.
    hdf_filename = str(config.RAW_DATA_DIR / 'flies_varwind_C1WT_combined_optotrigger_fancontrol_trimmed.parquet')
    df = pd.read_parquet(hdf_filename)

    # How many trajectories to process (max)
    # Could set to all of them if desired
    n_trajecs = 700

    # PARAMETERS
    obj_id_key='obj_id_unique_event'
    abs_min_ix=120   # min index (relative to trajec start) to use for roi
    abs_max_ix=500   # max index (relative to trajec start) to use for roi
    min_ix_range=190 # min number of indices to use for fit
    max_ix_range=210 # max number of indices to use for fit
    npoints=50                # num points for MIOP
    n_bootstraps=10           # number of bootstraps per trajec
    use_cvx_affine=True       # True: symmetric A; False: use pinv
    include_translation=args.include_translation # Include translation in the affine transform

    translation_flag = f'translation{include_translation}'


    wind_speeds = df.fan_speed_percent.dropna().unique()


    # FLASH
    if 0:
        for wind_speed in wind_speeds:
            print(wind_speed)
            braid_df = df[(df.fan_speed_percent==wind_speed) & (df.intensity>0)]
            
            unifying_algo_data = uaa.run_miop_affine_fit_on_trajectories(braid_df,    
                                                                         n_trajecs,  
                                                                         obj_id_key=obj_id_key,
                                                                         abs_min_ix=abs_min_ix,  
                                                                         abs_max_ix=abs_max_ix,   
                                                                         min_ix_range=min_ix_range, 
                                                                         max_ix_range=max_ix_range, 
                                                                         npoints=npoints,                
                                                                         n_bootstraps=n_bootstraps,         
                                                                         use_cvx_affine=use_cvx_affine,      
                                                                         include_translation=include_translation)
            unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'varwind_N{n_trajecs}_flash_{wind_speed}_{translation_flag}.parquet'))
        
    # SHAM
    braid_df = df[df.intensity==0]
    for wind_speed in wind_speeds:
        print(wind_speed)
        braid_df = df[(df.fan_speed_percent==wind_speed) & (df.intensity==0)]
        
        unifying_algo_data = uaa.run_miop_affine_fit_on_trajectories(braid_df,    
                                                                     n_trajecs,  
                                                                     obj_id_key=obj_id_key,
                                                                     abs_min_ix=abs_min_ix,  
                                                                     abs_max_ix=abs_max_ix,   
                                                                     min_ix_range=min_ix_range, 
                                                                     max_ix_range=max_ix_range, 
                                                                     npoints=npoints,                
                                                                     n_bootstraps=n_bootstraps,         
                                                                     use_cvx_affine=use_cvx_affine,      
                                                                     include_translation=include_translation)
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'varwind_N{n_trajecs}_sham_{wind_speed}_{translation_flag}.parquet'))


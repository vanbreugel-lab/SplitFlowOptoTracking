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

    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on unsteady experiment top-on wildtype data.')
    parser.add_argument('--include-translation', action='store_true', default=False,
                        help='Include translation in the affine transform (default: False)')
    args = parser.parse_args()

    n_trajecs = 200

    top_on_df = pd.read_parquet(str(config.RAW_DATA_DIR / 'flies_splitflow_topon_wt_preprocessed_optotrigger_trimmed.parquet'))

    df_flash_top_on = top_on_df[top_on_df.duration==675]
    df_sham_top_on = top_on_df[top_on_df.duration==0]

    print('All zones: ')
    print('Num trajecs flash: ' + str( len(df_flash_top_on.obj_id_unique_event.unique())))
    print('Num trajecs sham: ' + str( len(df_sham_top_on.obj_id_unique_event.unique())))

    #

    #flash_low_top_on = df_flash_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.median()<0.35)
    flash_low_top_on = df_flash_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.min()<0.26)
    flash_low_top_on = flash_low_top_on[~flash_low_top_on.xvel.isna()]

    flash_up_top_on  = df_flash_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.min()>0.26)
    flash_up_top_on = flash_up_top_on[~flash_up_top_on.xvel.isna()]

    #

    #sham_low_top_on = df_sham_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.median()<0.35)
    sham_low_top_on = df_sham_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.min()<0.26)
    sham_low_top_on = sham_low_top_on[~sham_low_top_on.xvel.isna()]

    sham_up_top_on  = df_sham_top_on.groupby("obj_id_unique_event").filter(lambda d: d.z.min()>0.26)
    sham_up_top_on = sham_up_top_on[~sham_up_top_on.xvel.isna()]

    print('Unsteady zone: ')
    print('Num trajecs flash: ' + str( len(flash_low_top_on.obj_id_unique_event.unique())))
    print('Num trajecs sham: ' + str( len(sham_low_top_on.obj_id_unique_event.unique())))

    print('Steady zone: ')
    print('Num trajecs flash: ' + str( len(flash_up_top_on.obj_id_unique_event.unique())))
    print('Num trajecs sham: ' + str( len(sham_up_top_on.obj_id_unique_event.unique())))

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

    if 0:
        # Run analysis on FLASH data LOW FLIES (unsteady)
        braid_df = flash_low_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'unsteady_top_on_data_WT_flash_{translation_flag}.parquet'))


        # Run analysis on SHAM data HIGH FLIES (unsteady)
        braid_df = sham_low_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'unsteady_top_on_data_WT_sham_{translation_flag}.parquet'))


    if 0:
        braid_df = flash_up_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'steady_top_on_data_WT_flash_{translation_flag}.parquet'))


        # Run analysis on SHAM data HIGH FLIES (unsteady)
        braid_df = sham_up_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'steady_top_on_data_WT_sham_{translation_flag}.parquet'))

    if 1:
        # Run analysis on FLASH data flies that dipped into unsteady region
        braid_df = flash_low_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'experiencedunsteady_top_on_data_WT_flash_{translation_flag}.parquet'))


        # Run analysis on SHAM data flies that dipped into unsteady region
        braid_df = sham_low_top_on
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
        unifying_algo_data.to_parquet(str(config.PREPROCESSED_DATA_DIR / f'experiencedunsteady_top_on_data_WT_sham_{translation_flag}.parquet'))



        # braid_df = flash_up_top_on
        # all_unifying_algo_data = run_miop_affine_fit_on_trajectories(braid_df, n_trajecs)
        # all_unifying_algo_data.to_parquet('steady_top_on_data_flash.parquet')

        # braid_df = sham_up_top_on
        # all_unifying_algo_data = run_miop_affine_fit_on_trajectories(braid_df, n_trajecs)
        # all_unifying_algo_data.to_parquet('steady_top_on_data_sham.parquet')



        # braid_df = flash_up_top_on
        # all_unifying_algo_data = run_miop_affine_fit_on_trajectories(braid_df, n_trajecs)
        # all_unifying_algo_data.to_parquet('steady_top_on_data_flash.parquet')

        # braid_df = sham_up_top_on
        # all_unifying_algo_data = run_miop_affine_fit_on_trajectories(braid_df, n_trajecs)
        # all_unifying_algo_data.to_parquet('steady_top_on_data_sham.parquet')

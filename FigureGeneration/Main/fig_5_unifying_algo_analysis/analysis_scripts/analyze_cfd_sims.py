import argparse
import pandas as pd

from align_course_direction_analysis import unifying_algo_analysis as uaa
from align_course_direction_analysis import unifying_algo_plots as uap

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on CFD simulation data.')
    parser.add_argument('--include-translation', action='store_true', default=False,
                        help='Include translation in the affine transform (default: False)')
    args = parser.parse_args()

    # Load the data
    hdf_filename =  '../trajectory_data/flies_laminar_c1xwt_preprocessed_optotrigger_trimmed.hdf'
    df_laminar = pd.read_hdf(hdf_filename)

    # How many trajectories to process (max)
    # Could set to all of them if desired
    n_trajecs = 200

    # PARAMETERS
    obj_id_key='obj_id_unique_event'
    abs_min_ix=120   # min index (relative to trajec start) to use for roi
    abs_max_ix=499   # max index (relative to trajec start) to use for roi
    min_ix_range=190 # min number of indices to use for fit
    max_ix_range=210 # max number of indices to use for fit
    npoints=50                # num points for MIOP
    n_bootstraps=10           # number of bootstraps per trajec
    use_cvx_affine=True       # True: symmetric A; False: use pinv
    include_translation=args.include_translation # Include translation in the affine transform

    translation_flag = f'translation{include_translation}'

    parquet_filename = '../trajectory_data/cfd_sim_circling_casting_all_wind_conditions_preprocessed.parquet'
    df_sims = pd.read_parquet(parquet_filename)

    # drop columns with nans
    df_sims = df_sims.dropna(axis=1)

    sim_exps = df_sims.dataset_name.unique()

    for sim_exp in sim_exps:
        print(sim_exp)
        braid_df = df_sims[df_sims.dataset_name==sim_exp]
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
        unifying_algo_data.to_parquet('../unifying_algo_results/' + 'cfd_sim_' + sim_exp.replace(' ', '_') + f'_{translation_flag}.parquet')

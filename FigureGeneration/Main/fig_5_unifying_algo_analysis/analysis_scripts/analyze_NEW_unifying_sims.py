import argparse
import pandas as pd

from align_course_direction_analysis import unifying_algo_analysis as uaa
from align_course_direction_analysis import unifying_algo_plots as uap

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run unifying algorithm analysis on unifying algo data.')
    parser.add_argument('--include-translation', action='store_true', default=False,
                        help='Include translation in the affine transform (default: False)')
    args = parser.parse_args()

    # Load the data
    hdf_filename =  '../trajectory_data/new_unifying.parquet'
    df = pd.read_parquet(hdf_filename)

    # How many trajectories to process (max)
    # Could set to all of them if desired
    n_trajecs = 200

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


    sim_exps = df.windtype.unique()

    for sim_exp in sim_exps:
        print(sim_exp)
        braid_df = df[df.windtype==sim_exp]
        
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
        unifying_algo_data.to_parquet(f'../unifying_algo_results/new_unifying_{sim_exp}_{translation_flag}.parquet')


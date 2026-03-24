import pandas as pd

from align_course_direction_analysis import unifying_algo_analysis as uaa
from align_course_direction_analysis import unifying_algo_plots as uap

if __name__ == '__main__':

    # Load the data
    hdf_filename =  'MatchedOrco_chrimson_trim.parquet'
    braid_df = pd.read_parquet(hdf_filename)

    # How many trajectories to process (max)
    # Could set to all of them if desired
    n_trajecs = 100

    # PARAMETERS
    obj_id_key='obj_id_unique_event'
    abs_min_ix = 2000
    abs_max_ix = 3000
    min_ix_range = 400
    max_ix_range = 500
    npoints=50                # num points for MIOP
    n_bootstraps=10           # number of bootstraps per trajec
    use_cvx_affine=True       # True: symmetric A; False: use pinv
    include_translation=True # Include translation in the affine transform

    pis_max = 20
    intercept_max = 4
    huber_M=1.0
    ignore_large_errors=True
    error_threshold=2.0
    large_error_penalty=1.0

    # Run analysis on FLASH data
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
                                                                 include_translation=include_translation, 
                                                                 pis_max = pis_max, 
                                                                 intercept_max = intercept_max,
                                                                 huber_M=huber_M,
                                                                 ignore_large_errors=ignore_large_errors,
                                                                 error_threshold=error_threshold,
                                                                 large_error_penalty=large_error_penalty)
    unifying_algo_data.to_parquet('unifying_algo_results_MatchedOrco_chrimson_trim.parquet')



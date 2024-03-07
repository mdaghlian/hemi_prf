#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

## qsub eg...
import sys
import getopt
import os
opj = os.path.join
import time

import numpy as np
import scipy.io as sio

from prfpy_csenf.model import *
from prfpy_csenf.fit import *

from dag_prf_utils.utils import * 
from dag_prf_utils.prfpy_functions import *

from hemi_prf.utils import *
from hemi_prf.load_functions import *  

derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/hemi_prf/derivatives'
prf_dir = opj(derivatives_dir, 'prf')
if not os.path.exists(prf_dir):
    os.mkdir(prf_dir)    

def main(argv):
    '''
    ---------------------------
    Run the fitting procedure

    Args (optional):
        --sub               subject number
        --ses               session number
        --model             model (css, dog, norm)
        --n_jobs <n_jobs>   number of jobs to run in parallel
        --fit_hrf True      whether to fit the HRF or not, False if mrVista HRF is used
        --grid_nr <grid_nr> which grid to use & number...           
        --roi_fit           mask by roi
        --tc                use the TC constraints
        --bgfs              use the bgfs constraints        
        --skip_iter         skip the iterative fitting stage
        -h          help
    
nr_jobs=20    
job_name=hprf    
qsub -q short.q@jupiter -pe smp $nr_jobs -wd $PWD -N $job_name -o $job_name.txt analysis_steps/s003_gaussian_fit.py --sub sub-01 --n_jobs 20
    '''
    # Load the fitting settings, add in the new info
    fit_settings = hprf_load_settings_file()   # take defaults from the yaml file 
    # [1] Setup parameters:
    sub = None
    ses = 'ses-1'
    n_jobs = fit_settings['n_jobs']
    fit_hrf = fit_settings['fit_hrf']
    roi_fit = 'all'
    constraint_type = fit_settings['constraint_type']
    grid_nr = fit_settings['grid_nr']
    skip_iter = fit_settings['skip_iter']
    detrend = int(0)

    try:
        opts = getopt.getopt(argv,"h:s:",["sub=", "ses=", "model=", "n_jobs=","roi_fit=", "fit_hrf=", "grid_nr=", "detrend=", "tc", "bgfs", "skip_iter"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)    

    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
            sys.exit()

        elif opt in ("--sub"):
            sub = dag_hyphen_parse('sub', arg)
        elif opt in ("--ses"):
            ses = dag_hyphen_parse('ses', arg) 
        elif opt in ("--model"):
            model = arg
        elif opt in ("--n_jobs"):
            n_jobs = int(arg)
        elif opt in ("--fit_hrf"):
            fit_hrf = bool(int(arg))
        elif opt in ("--roi_fit"):
            roi_fit = arg
            print(roi_fit)
        elif opt in ("--grid_nr"):
            grid_nr = int(arg)
        elif opt in ("--detrend"):
            detrend = int(arg)
        elif opt in ("--tc"):
            constraint_type = 'tc'
        elif opt in ("--bgfs"):
            constraint_type = 'bgfs'
        elif opt in ("--skip_iter"):
            skip_iter = True
            constraint_type = 'skip'

    if constraint_type == 'tc':
        constraints = []
    elif constraint_type == 'bgfs':
        constraints = None

    # If changed, overwrite the settings
    fit_settings['model'] = model
    fit_settings['n_jobs'] = n_jobs
    fit_settings['fit_hrf'] = fit_hrf
    fit_settings['grid_nr'] = grid_nr
    fit_settings['roi_fit'] = roi_fit
    fit_settings['constraint_type'] = constraint_type
    fit_settings['skip_iter'] = skip_iter    
    fit_settings['detrend'] = detrend
    # Path to prf
    sub_prf_dir = opj(prf_dir, sub, ses)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(sub_prf_dir):
        os.makedirs(sub_prf_dir)    
    
    # Get ready to save everything:
    # make save name saying what we did
    if fit_hrf:
        hrf_str = 'fit'
    else:
        hrf_str = 'no'    
    
    iter_prf_name = f'{sub}_{ses}_task-prf_model-{model}_hrf-{hrf_str}_roi-{roi_fit}_detrend-{detrend}_stage-iter-{constraint_type}.pkl'
    grid_prf_name = f'{sub}_{ses}_task-prf_model-{model}_hrf-{hrf_str}_roi-{roi_fit}_detrend-{detrend}_stage-grid.pkl'

    # Load the timeseries 
    # [1] Load the data:
    # -> is it in the whole_brain_ts folder?
    ts_psc = hprf_load_ts_psc(sub, ses, detrend=detrend)
    # Prfpy stim
    prf_stim = hprf_load_prfpy_stim() 
    fit_settings['max_ecc'] = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region          
    n_vx = ts_psc.shape[0]
    # Also exclude mean std=0 voxels
    vx_mask = hprf_load_roi(sub, roi_fit)        
    vx_mask &= ts_psc.std(axis=1) > 0.1

    print(f'Number of vertices total: {n_vx}')
    print(f'Number of vertices in mask (roi={roi_fit}), std != 0: {vx_mask.sum()}')

    # Setup bounds:
    gauss_bounds = {
        'x'             : (-1.5*fit_settings['max_ecc'], 1.5*fit_settings['max_ecc']),          # x bound
        'y'             : (-1.5*fit_settings['max_ecc'], 1.5*fit_settings['max_ecc']),          # y bound
        'size_1'        : (1e-1, fit_settings['max_ecc']*3),                             # prf size bounds
        'amp_1'         : (fit_settings['gauss_bounds']['amp_1'][0],fit_settings['gauss_bounds']['amp_1'][1]),      # prf amplitude
        'bold_baseline' : (fit_settings['gauss_bounds']['bold_baseline'][0],fit_settings['gauss_bounds']['bold_baseline'][1]),      # bold baseline (fixed)
    }
    if fit_hrf:
        gauss_bounds['hrf_1'] = (fit_settings['hrf']['deriv_bound'][0], fit_settings['hrf']['deriv_bound'][1]), # hrf_1 bound
        gauss_bounds['hrf_2'] = (fit_settings['hrf']['disp_bound'][0],  fit_settings['hrf']['disp_bound'][1]), # hrf_2 bound
    else:
        hrf_1_val = fit_settings['hrf']['pars'][1]
        hrf_2_val = fit_settings['hrf']['pars'][2]
        gauss_bounds['hrf_1'] = (hrf_1_val,hrf_1_val)
        gauss_bounds['hrf_2'] = (hrf_2_val,hrf_2_val)

    
    # *********************************************************************
    # ********* ********* MAKE GAUSSIAN MODEL ********* ********* *********
    gg = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf = fit_settings['hrf']['pars'],
        )
    # Make fitter object
    gf = Iso2DGaussianFitter(
        data=ts_psc[vx_mask,:],             # time series
        model=gg,                       # model (see above)
        n_jobs=n_jobs,
        )
    # -> load in the previous parameters
    try: # Try the roi specific one
        gauss_fit_params = hprf_load_fit_prfs(
            sub=sub, ses=ses, model='gauss', detrend=detrend, include=[f'roi-{roi_fit}', constraint_type, f'hrf-{hrf_str}']
        )
    except: # Otherwise load the 'all'
        gauss_fit_params = hprf_load_fit_prfs(
            sub=sub, ses=ses, model='gauss', detrend=detrend, include=['roi-all', constraint_type, f'hrf-{hrf_str}']
        )        

    gf.iterative_search_params = gauss_fit_params[vx_mask,:-1].copy()
    print(f'Gaussian fit: number of parameter= {gf.iterative_search_params.shape}')
    gf.rsq_mask = gauss_fit_params[vx_mask,-1] > fit_settings['rsq_threshold']

    # CREATE BOUNDS ALL MODELS SHARE (i.e., gaussian)
    common_iter_bounds = [ # For the iterative fit
        (gauss_bounds['x']),
        (gauss_bounds['y']),
        (gauss_bounds['size_1']),
        (gauss_bounds['amp_1']),   
        (gauss_bounds['bold_baseline']),   
    ]

    hrf_bounds = [
        (gauss_bounds['hrf_1']),
        (gauss_bounds['hrf_2']),
    ]

    # if fit_hrf:
    #     hrf_bounds = [
    #         (gauss_bounds['hrf_1']),
    #         (gauss_bounds['hrf_2']),
    #     ]
    # else:
    #     hrf_bounds = []

    # *********************************************************************
    # Next make the model specific fitters & bounds
    if model=='norm': # ******************************** NORM
        gg_ext = Norm_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf = fit_settings['hrf']['pars'],
            )
        gf_ext = Norm_Iso2DGaussianFitter(
            data=ts_psc[vx_mask,:],           
            model=gg_ext,                  
            n_jobs=n_jobs,
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = fit_settings['use_previous_gaussian_fitter_hrf'], 
            )
        # Extra grid bounds
        ext_grid_bounds = [
            fit_settings['gauss_bounds']['amp_1'],
            fit_settings['norm']['d_val_bound'],
        ]
        # Extra grid values
        ext_grids = [
            np.array(fit_settings['norm']['amp_2_grid'], dtype='float32'),
            np.array(fit_settings['norm']['size_2_grid'], dtype='float32'),
            np.array(fit_settings['norm']['b_val_grid'], dtype='float32'),
            np.array(fit_settings['norm']['d_val_grid'], dtype='float32'),            
        ]
        # Extra iterative bounds
        ext_iter_bounds = [
            (fit_settings['norm']['amp_2_bound']),
            (1e-1, fit_settings['max_ecc']*6),   
            (fit_settings['norm']['b_val_bound']), 
            (fit_settings['norm']['d_val_bound']),
            ] 
        
    elif model=='dog': # ******************************** DOG
        gg_ext = DoG_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf = fit_settings['hrf_pars'],
            )
        gf_ext = DoG_Iso2DGaussianFitter(
            data=ts_psc[vx_mask,:],                   
            model=gg_ext,                  
            n_jobs=n_jobs,
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = fit_settings['use_previous_gaussian_fitter_hrf'], 
            )
        # Extra grid bounds
        ext_grid_bounds = [
            fit_settings['gauss_bounds']['amp_1'],
            fit_settings['dog']['amp_2_bound'],
        ]
        # Extra grid values
        ext_grids = [
            np.array(fit_settings['dog']['amp_2_grid'], dtype='float32'),
            np.array(fit_settings['dog']['size_2_grid'], dtype='float32'),
        ]
        # Extra iterative bounds
        ext_iter_bounds = [
            (fit_settings['dog']['amp_2_bound']),
            (1e-1, fit_settings['max_ecc']*6),   
            ]

    elif model=='css': # ******************************** CSS
        gg_ext = CSS_Iso2DGaussianModel(
            stimulus=prf_stim,                                  
            hrf = fit_settings['hrf']['pars'],
            )
        gf_ext = CSS_Iso2DGaussianFitter(
            data=ts_psc[vx_mask,:],                  
            model=gg_ext,                  
            n_jobs=n_jobs,
            previous_gaussian_fitter = gf,
            use_previous_gaussian_fitter_hrf = fit_settings['use_previous_gaussian_fitter_hrf'], 
            )
        # Extra grid bounds
        ext_grid_bounds = [fit_settings['gauss_bounds']['amp_1']]
        # Extra grid values
        ext_grids = [
            np.array(fit_settings['css']['n_exp_grid'], dtype='float32'),
        ]

        ext_iter_bounds = [
            (fit_settings['css']['n_exp_bound']),  # css exponent 
            ]

    # Combine the bounds 
    full_iter_bounds = common_iter_bounds.copy() + ext_iter_bounds.copy() + hrf_bounds.copy()
    # DO GRID FIT
    # Time how long the fitting takes
    start_time = time.time()    
    print(f'Starting grid')
    #        
    gf_ext.grid_fit(
        *ext_grids,
        verbose=True,
        n_batches=fit_settings['n_jobs'],
        rsq_threshold=fit_settings['rsq_threshold'],
        fixed_grid_baseline=fit_settings['fixed_grid_baseline'],
        grid_bounds=ext_grid_bounds,
    )

    # Fiter for nans
    gf_ext.gridsearch_params = dag_filter_for_nans(gf_ext.gridsearch_params)
    # Print how long it took in minutes and seconds
    grid_time = (time.time() - start_time)/60
    print(f'Grid fitting took {grid_time} minutes')    
    vx_gt_rsq_th = gf_ext.gridsearch_params[:,-1]>fit_settings['rsq_threshold']
    nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
    mean_vx_gt_rsq_th = np.mean(gf_ext.gridsearch_params[vx_gt_rsq_th,-1])
    print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')
        
    #  DO ITERATIVE FIT
    if not skip_iter:
        # Start iterative fit    
        print('Starting iterative fit')
        print(gf_ext.gridsearch_params.shape)
        print(len(full_iter_bounds))

        gf_ext.iterative_fit(
            rsq_threshold = fit_settings['rsq_threshold'],
            verbose = False,
            bounds = full_iter_bounds,
            constraints = constraints,
            xtol=fit_settings['xtol'],   
            ftol=fit_settings['ftol'],           
            )
        gf_ext.iterative_search_params = dag_filter_for_nans(gf_ext.iterative_search_params)          
        # Print how long it took in minutes and seconds
        iter_time = (time.time() - start_time)/60
        iter_time = iter_time - grid_time
        print(f'Iterative fitting took {iter_time} minutes')
        print(f'Total time = {iter_time + grid_time} minutes')
        m_rsq = gf_ext.iterative_search_params[:,-1].mean()
        rsq_gt_pt1 = np.sum(gf_ext.iterative_search_params[:,-1]>0.1)
        m_rsq_th = gf_ext.iterative_search_params[gf_ext.iterative_search_params[:,-1]>0.1,-1].mean()
        print(f'Iterative stage, of voxels fit: Mean r^2 = {m_rsq}')
        print(f'Iterative stage, nr voxels with rsq>0.1={rsq_gt_pt1}')
        print(f'Iterative stage, mean rsq for voxels with rsq>0.1={m_rsq_th}')    
    else:
        print(f'Skipping iterative fit')
        iter_time = 0
        gf_ext.iterative_search_params = gf_ext.gridsearch_params.copy()        

    print(gf_ext.iterative_search_params.shape)

    iter_params = np.zeros((n_vx, len(full_iter_bounds) + 1))
    iter_params[vx_mask,:] = gf_ext.iterative_search_params        

    # Save the rest of the stuff:    
    fit_settings['full_iter_bounds'] = full_iter_bounds
    fit_settings['ext_grid_bounds'] = ext_grid_bounds
    fit_settings['ext_grids'] = ext_grids
    fit_settings['grid_time'] = grid_time
    fit_settings['iter_time'] = iter_time

    # Save everything in a pickle
    pkl_dict = {}
    pkl_dict['settings'] = fit_settings
    pkl_dict['pars'] = iter_params

    pkl_file = opj(sub_prf_dir, iter_prf_name)
    print(f'Saving {pkl_file}')
    # Save as pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(pkl_dict, f)
    print('DONE!!!!!!!!!!!')




if __name__ == "__main__":
    main(sys.argv[1:])    
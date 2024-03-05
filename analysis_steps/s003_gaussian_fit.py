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

from prfpy.model import *
from prfpy.fit import *

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

    try:
        opts = getopt.getopt(argv,"h:s:",["sub=", "n_jobs=","roi_fit=", "fit_hrf=", "grid_nr=", "detrend=", "tc", "bgfs", "skip_iter"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)    

    for opt, arg in opts:
        if opt == '-h':
            print(main.__doc__)
            sys.exit()

        elif opt in ("s", "--sub"):
            sub = f'sub-{arg}'        
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
        gauss_constraints = []
    elif constraint_type == 'bgfs':
        gauss_constraints = None

    # If changed, overwrite the settings
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
    bounds = {
        'x' : (-1.5*fit_settings['max_ecc'], 1.5*fit_settings['max_ecc']),          # x bound
        'y' : (-1.5*fit_settings['max_ecc'], 1.5*fit_settings['max_ecc']),          # y bound
        'size_1' : (1e-1, fit_settings['max_ecc']*3),                             # prf size bounds
        'amp_1' : (fit_settings['prf_ampl'][0],fit_settings['prf_ampl'][1]),      # prf amplitude
        'bold_baseline' : (fit_settings['bold_bsl'][0],fit_settings['bold_bsl'][1]),      # bold baseline (fixed)
        'hrf_1' : (fit_settings['hrf']['deriv_bound'][0], fit_settings['hrf']['deriv_bound'][1]), # hrf_1 bound
        'hrf_2' : (fit_settings['hrf']['disp_bound'][0],  fit_settings['hrf']['disp_bound'][1]), # hrf_2 bound
    }
    # -> & grid bounds
    gauss_grid_bounds = [[fit_settings['prf_ampl'][0],fit_settings['prf_ampl'][1]]] 


    # [2] Setup grids:
    # -> grids
    ecc_grid    = fit_settings['max_ecc'] * np.linspace(0.25, 1, grid_nr)**2 # Squared because of cortical magnification, more efficiently tiles the visual field...
    size_grid   = fit_settings['max_ecc'] * np.linspace(0.1, 1, grid_nr)**2  # Possible size values (i.e., sigma in gaussian model) 
    polar_grid  = np.linspace(0, 2*np.pi, grid_nr)              # Possible polar angle coordinates

    if fit_hrf:
        hrf_1_grid = np.linspace(bounds['hrf_1'][0], bounds['hrf_1'][1], grid_nr)
        hrf_2_grid = np.array(bounds['hrf_2'][0]) # don't fit hrf_2
        # [3] Setup iterative bounds
        bounds_list = [
            bounds['x'],
            bounds['y'],
            bounds['size_1'],
            bounds['amp_1'],   
            bounds['bold_baseline'],      
            bounds['hrf_1'],      
            bounds['hrf_2'],      
        ]        

    else:
        # Override the HRF
        hrf_1_grid = None
        hrf_2_grid = None
        # -> &  bounds
        bounds_list = [
            bounds['x'],
            bounds['y'],
            bounds['size_1'],
            bounds['amp_1'],   
            bounds['bold_baseline'],   
            (1,1),      
            (0,0),               
        ]

    # Get ready to save everything:
    # make save name saying what we did
    if fit_hrf:
        hrf_str = 'fit'
    else:
        hrf_str = 'no'    
    
    iter_prf_name = f'{sub}_{ses}_task-prf_model-gauss_hrf-{hrf_str}_roi-{roi_fit}_detrend-{detrend}_stage-iter-{constraint_type}.pkl'
    grid_prf_name = f'{sub}_{ses}_task-prf_model-gauss_hrf-{hrf_str}_roi-{roi_fit}_detrend-{detrend}_stage-grid.pkl'
    

    # Make the prfpy model
    gauss_model = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf = fit_settings['hrf']['pars'],
        )
    # Time how long the fitting takes
    start_time = time.time()

    # Make fitter object
    gauss_fitter = Iso2DGaussianFitter(
        data=ts_psc[vx_mask,:],             # time series
        model=gauss_model,                       # model (see above)
        n_jobs=n_jobs,
        )    

    # Start grid fit
    gauss_fitter.grid_fit(
        ecc_grid=ecc_grid,
        polar_grid=polar_grid,
        size_grid=size_grid,
        hrf_1_grid=hrf_1_grid,
        hrf_2_grid=hrf_2_grid,
        verbose=True,
        n_batches=n_jobs,                          # The grid fit is performed in parallel over n_batches of units.Batch parallelization is faster than single-unit parallelization and of sequential computing.
        fixed_grid_baseline=fit_settings['fixed_grid_baseline'],    # Fix the baseline? This makes sense if we have fixed the baseline in preprocessing
        grid_bounds=gauss_grid_bounds
        )
    # Print how long it took in minutes and seconds
    grid_time = (time.time() - start_time)/60
    print(f'Grid fitting took {grid_time} minutes')
    m_rsq = gauss_fitter.gridsearch_params[:,-1].mean()
    rsq_gt_pt1 = np.sum(gauss_fitter.gridsearch_params[:,-1]>0.1)
    m_rsq_th = gauss_fitter.gridsearch_params[gauss_fitter.gridsearch_params[:,-1]>0.1,-1].mean()
    print(f'Grid stage, of voxels fit: Mean r^2 = {m_rsq}')
    print(f'Grid stage, nr voxels with rsq>0.1={rsq_gt_pt1}')
    print(f'Grid stage, mean rsq for voxels with rsq>0.1={m_rsq_th}')

    # Save the grid fit
    # print(f'Saving {grid_prf_name}')
    grid_params = np.zeros((n_vx, len(bounds_list)+1))
    grid_params[vx_mask,:] = gauss_fitter.gridsearch_params    
    # Save everything in a pickle
    pkl_dict = {}
    pkl_dict['pars'] = grid_params
    pkl_file = opj(sub_prf_dir, grid_prf_name)
    print(f'Saving {pkl_file}')
    # Save as pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(pkl_dict, f)    

    if not skip_iter:
        # Start iterative fit
        print('Starting iterative fit')
        gauss_fitter.iterative_fit(
            rsq_threshold = fit_settings['rsq_threshold'],
            verbose = True,
            bounds = bounds_list,
            constraints = gauss_constraints,
            xtol=float(fit_settings['xtol']),   
            ftol=float(fit_settings['ftol']),           
            )
        # Print how long it took in minutes and seconds
        iter_time = (time.time() - start_time)/60
        iter_time = iter_time - grid_time
        print(f'Iterative fitting took {iter_time} minutes')
        print(f'Total time = {iter_time + grid_time} minutes')
        m_rsq = gauss_fitter.iterative_search_params[:,-1].mean()
        rsq_gt_pt1 = np.sum(gauss_fitter.iterative_search_params[:,-1]>0.1)
        m_rsq_th = gauss_fitter.iterative_search_params[gauss_fitter.iterative_search_params[:,-1]>0.1,-1].mean()
        print(f'Iterative stage, of voxels fit: Mean r^2 = {m_rsq}')
        print(f'Iterative stage, nr voxels with rsq>0.1={rsq_gt_pt1}')
        print(f'Iterative stage, mean rsq for voxels with rsq>0.1={m_rsq_th}')    
    else:
        print(f'Skipping iterative fit')
        iter_time = 0
        gauss_fitter.iterative_search_params = gauss_fitter.gridsearch_params.copy()        

    # Use the mask to plug back in the new fit values    
    iter_params = np.zeros((n_vx, len(bounds_list) + 1))
    iter_params[vx_mask,:] = gauss_fitter.iterative_search_params        
    print(iter_params.shape)

    # Save the rest of the stuff:
    fit_settings['bounds_list'] = bounds_list
    fit_settings['gauss_grid_bounds'] = gauss_grid_bounds
    fit_settings['ecc_grid'] = ecc_grid
    fit_settings['polar_grid'] = polar_grid
    fit_settings['size_grid'] = size_grid
    fit_settings['hrf_1_grid'] = hrf_1_grid
    fit_settings['hrf_2_grid'] = hrf_2_grid
    fit_settings['grid_time'] = grid_time
    fit_settings['iter_time'] = iter_time

    # Save everything in a pickle
    pkl_dict = {}
    pkl_dict['settings'] = fit_settings
    pkl_dict['pars'] = iter_params
    # pkl_dict['preds'] = preds

    pkl_file = opj(sub_prf_dir, iter_prf_name)
    print(f'Saving {pkl_file}')
    # Save as pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(pkl_dict, f)
    print('DONE!!!!!!!!!!!')

if __name__ == "__main__":
    main(sys.argv[1:])    
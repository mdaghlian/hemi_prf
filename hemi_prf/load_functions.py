import os
import h5py
opj = os.path.join
import numpy as np
from prfpy.stimulus import PRFStimulus2D
import yaml
import pickle
from hemi_prf.utils import *
from dag_prf_utils.utils import dag_find_file_in_folder, dag_load_roi

path_to_code = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

path_saved = opj(path_to_code, 'saved_data')
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/hemi_prf/derivatives' 
ts_dir = opj(derivatives_dir, 'surface_timeseries_npy')
fs_dir = opj(derivatives_dir, 'freesurfer')

# ****************************************************
# FUNCTIONS TO LOAD PRF INFORMATION 
def hprf_load_settings_file():
    '''
    Load the settings in the .yml file    
    '''
    prf_settings_file = opj(path_saved, 'fit_settings_prf.yml')
    with open(prf_settings_file) as f:
        prf_settings = yaml.safe_load(f)    
    return prf_settings
    
def hprf_load_dm_mat():
    '''
    Load the matlab design matrix
    '''
    dm_file = opj(path_saved, 'aps_hemiamap.mat')
    f =  h5py.File(dm_file, 'r')
    for k,v in f.items():
        dm = np.array(v)
    dm = np.moveaxis(dm, 0, -1)  
    return dm      

def hprf_load_prfpy_stim():
    dm = hprf_load_dm_mat()
    prf_info = hprf_load_settings_file()
    prf_stim = PRFStimulus2D(
        screen_size_cm=prf_info['screen_size_cm'],          # Distance of screen to eye
        screen_distance_cm=prf_info['screen_distance_cm'],  # height of the screen (i.e., the diameter of the stimulated region)
        design_matrix=dm,                                   # dm (npix x npix x time_points)
        TR=prf_info['TR'],                                  # TR
        )    
    return prf_stim

def hprf_load_roi(sub, roi):
    roi = dag_load_roi(sub, roi, fs_dir=fs_dir)
    return roi
# ****************************************************
# FUNCTIONS TO LOAD OUTPUT OF gii_to_numpy (i.e., timeseries)
def hprf_load_ts_psc(sub, ses='ses-1', detrend=False):
    '''
    Load timeseries both hemispheres
    PSC (percent signal change)
    The output of "gii_to_numpy"
    '''
    sub_ts_dir = opj(ts_dir, sub, ses) 
    file = dag_find_file_in_folder(
        filt=[sub, '_lr_psc.npy',  f'detrend-{int(detrend)}'],
        path=sub_ts_dir, 
    )
    return np.load(file)

def hprf_load_ts_NOT_psc(sub, ses='ses-1'):
    '''
    Load timeseries both hemispheres
    BEFORE PSC (but still the mean)
    The output of "gii_to_numpy"
    '''
    sub_ts_dir = opj(ts_dir, sub, ses) 
    file = dag_find_file_in_folder(
        filt=[sub, '_lr_raw.npy'],
        path=sub_ts_dir, 
    )
    return np.load(file)

def hprf_load_mean_epi(sub, ses='ses-1'):
    '''
    Load mean EPI. Useful for checking the cortical surface for veins
    Low mean EPI means that there might be a vein
    > The output of "gii_to_numpy"
    '''
    sub_ts_dir = opj(ts_dir, sub, ses) 
    file = dag_find_file_in_folder(
        filt=[sub, '_lr_mean_epi.npy'],
        path=sub_ts_dir, 
    )
    return np.load(file)

def hprf_load_run_correlation(sub, ses='ses-1', detrend=False):
    '''
    Load split half correlation. 
    Useful for checking what the maximum variance explained you'd expect
    > The output of "gii_to_numpy"
    '''
    sub_ts_dir = opj(ts_dir, sub, ses) 
    file = dag_find_file_in_folder(
        filt=[sub, '_lr_run_correlation.npy',  f'detrend-{int(detrend)}'],
        path=sub_ts_dir, 
    )
    return np.load(file)




# ****************************************************
# FUNCTIONS TO LOAD OUTPUT OF PRF FITTING

def hprf_load_fit_prfs(sub,ses='ses-1',  **kwargs):
    pkl_file = hprf_find_pkl_file(sub,ses=ses, **kwargs)
    prfpy_params = hprf_load_prf_pkl(pkl_file, key='pars')
    return prfpy_params

def hprf_load_fit_settings(sub,ses='ses-1', **kwargs):
    pkl_file = hprf_find_pkl_file(sub, ses=ses, **kwargs)
    prfpy_settings = hprf_load_prf_pkl(pkl_file, key='settings')
    return prfpy_settings

def hprf_find_pkl_file(sub,ses, **kwargs):
    '''
    Find the pkl file
    '''
    detrend = kwargs.get('detrend', 0)
    roi_fit = kwargs.get('roi_fit', 'all')  # default all
    hrf_fit = kwargs.get('hrf_fit', 'hrf-no')   # default mrvista hrf
    fit_stage = kwargs.get('fit_stage', 'iter')   # default mrvista hrf    
    model = kwargs.get('model', 'gauss')

    include = kwargs.get('include', [])
    include += [model, '.pkl', roi_fit, hrf_fit, fit_stage, f'detrend-{detrend}']
    # remove empty strings
    include = [x for x in include if x != '']    
    exclude = kwargs.get('exclude', []) # by default remove any detrend
    # Remove any matches between include & exclude lists
    # -> including partial matches. i.e., if part of a string in include is in exclude then remove it from exclude
    exclude = [x for x in exclude if not any([x in y for y in include])]        

    # [1] Load the data:
    sub_prf_dir = opj(derivatives_dir, 'prf', sub, ses)
    
    # -> get the list of files
    pkl_file = dag_find_file_in_folder(
        include,
        sub_prf_dir,
        exclude=exclude,
    )
    # If more than 1 file returned print them out and exit
    if isinstance(pkl_file, list):
        print('More than 1 file found:')
        for f in pkl_file:
            print(f.split('/')[-1])
        raise ValueError('More than 1 file found')
    return pkl_file

def hprf_load_prf_pkl(pkl_path, key=None):
    '''Load the pkl file'''
    pkl_file = open(pkl_path,'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()     
    if key is not None:
        data = data[key]
    return data   
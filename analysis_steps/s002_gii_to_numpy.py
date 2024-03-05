#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

## qsub eg...
import sys
import getopt
import os
opj = os.path.join

import numpy as np
import nibabel as nib

from dag_prf_utils.utils import *
from dag_prf_utils.stats import *

from hemi_prf.load_functions import *
from hemi_prf.utils import *


derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/hemi_prf/derivatives'

def main(argv):
    '''
    ---------------------------
    * find .gii files in a path which match a certain pattern
    * concatenate over hemispheres. save mean .npy
    * save mean_EPI files
    
    Args:
        --sub      subject
        --ses      session
        --detrend       [optional] number of trends to remove. Default is false (just does PSC)
    ---------------------------
    '''
    search_excl = None
    detrend  = False
    ses = 'ses-1'
    for i,arg in enumerate(argv):
        if '--sub' in arg:
            sub = argv[i+1]
        if '--ses' in arg:
            ses = argv[i+1]            
        elif '--detrend' in arg:
            detrend = int(argv[i+1])
    mean_name = f'{sub}_{ses}_mean-prf'
    # [1] Make the output folder
    npy_file = opj(derivatives_dir, 'surface_timeseries_npy')
    if not os.path.exists(npy_file):
        os.makedirs(npy_file)
    sub_npy_file = opj(npy_file, sub, ses)
    if not os.path.exists(sub_npy_file):
        os.makedirs(sub_npy_file)
    
    # [2] Select gii_file
    gii_file = opj(derivatives_dir, 'fmriprep', sub, ses, 'func')

    # [3] Create the include flag
    incl_flag = 'fsnative,task-' # both motion and retino
    
    # [4] Write the command to call dag_gii_to_npy
    cmd = f'dag_gii_to_npy --gii_file {gii_file} --npy_file {sub_npy_file} --mean_name {mean_name} --include {incl_flag} --detrend {int(detrend)}'

    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    main(sys.argv[1:])        
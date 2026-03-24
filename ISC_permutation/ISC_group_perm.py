#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 10:26:46 2025

@author: robfren
"""

import os
import numpy as np
from scipy import stats
from brainiak import isc

from brainiak.io import load_boolean_mask
import nibabel as nib

#%%

# group level permutation test - Hypothesis 1

thresh = 0.001
FDR = True

nperm = 10000

#%% split ver for memory

basepath = './ISC_subject_maps'

sublist = np.loadtxt('./sublist.txt', dtype='str')

grp = np.array([1 if int(i.split('-')[1])>1000 else 2 for i in sublist])

#%%


# load subject ISC into memmap


splits = 1
stepsize = 228483//splits

p_array = np.empty(228483)
group_diff_array = np.empty(228483)


for step in range(0,splits+1):
    print(f'running step {step} ...', end=' ')
    start =  step * stepsize
    stop = (step+1) * stepsize
    if stop>228483:
        stop = 228483
    
    
    allsub = np.empty((len(sublist), (stop-start)))
    
    for idx, sub in enumerate(sublist):
        sub_isc = np.load(f'{basepath}/{sub}.npy')
        allsub[idx, :] = sub_isc[start:stop]
    
    group_diff, p, _ = isc.permutation_isc(allsub, group_assignment=grp,
                                           pairwise=False,
                                           summary_statistic='median',
                                           n_permutations=nperm,
                                           side='two-sided')
        
    p_array[start:stop] = p
    group_diff_array[start:stop] = group_diff
    
    print('Done.')


#%%
if FDR:
    p_array = stats.false_discovery_control(p_array)

#%% save full arrays to disk
if not os.path.exists('./YAvOA_perm'):
    os.makedirs('./YAvOA_perm')

np.save('./YAvOA_perm/group_diff.npy', group_diff_array)
np.save('./YAvOA_perm/pval.npy', p_array)

# threshold median based on p significance

thresh_grp = np.where(p_array<thresh, group_diff_array, 0)
np.save(f'./YAvOA_perm/thresh_median_{thresh}.npy', thresh_grp)

#%% save as brain maps MNI space

mask = './misc/MNI152_T1_2mm_brain_mask.nii.gz'

# point to anatomical reference template- needs same affine as input.
anat_template = './misc/MNI152_T1_2mm.nii'
ref_nii = nib.load(anat_template)

mask_img = load_boolean_mask(mask)
mask_coords = np.where(mask_img)

for bmap, name in zip([group_diff_array, p_array, thresh_grp], ['group_diff', 'pval', f'thresh_group_diff_{thresh}']):

    # Create empty 3D image and populate
    # with thresholded ISC values
    isc_img = np.full(ref_nii.shape, np.nan)
    
    isc_img[mask_coords] = bmap
    
    # Convert to NIfTI image
    isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
    isc_fn = f'.YAvOA_perm/{name}_fdr-{FDR}.nii.gz'
    nib.save(isc_nii, isc_fn)

#%%


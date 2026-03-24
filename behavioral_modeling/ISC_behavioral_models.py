#%%
import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 12})

from nilearn import surface
from nilearn import plotting
from nilearn import datasets
from nilearn.surface import SurfaceImage

#%%

def plot_surf(axis_avg, contour_bin, model_name, contour_col = 'g', nilearn_kwargs = {}, sym_cbar=False, mesh_qual='high'):

    lh_v = axis_avg[:100]
    rh_v = axis_avg[100:]
    contour_bin_l = contour_bin[:100]
    contour_bin_r = contour_bin[100:]

    # this needs to point to local copy of freesurfer schaefer parcellation - get all 3 for different mesh qualities.
    # found here: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3
    bdir = '/local/path/to/schaefer/freesurfer/parcellation/FreeSurfer5.3'
    
    if mesh_qual.lower() == 'high':
        fs = 'fsaverage'
    elif mesh_qual.lower() == 'med':
        fs = 'fsaverage6'
    elif mesh_qual.lower() == 'low':
        fs = 'fsaverage5'
    else:
        raise ValueError('`mesh_qual` must be "high", "med", or "low"')

    schaefer_l_lab = surface.load_surf_data(f'{bdir}/{fs}/label/lh.Schaefer2018_200Parcels_17Networks_order.annot')
    schaefer_r_lab = surface.load_surf_data(f'{bdir}/{fs}/label/rh.Schaefer2018_200Parcels_17Networks_order.annot')

    # Retrieve fsaverage5 surface dataset for the plotting background. 
    fsaverage = datasets.fetch_surf_fsaverage(fs)
    fs_mapping = datasets.load_fsaverage(fs)

    # alter labels to reflect the collapsed regions
    mapping_l = {k: v for k, v in zip(range(1,101), lh_v)}
    mapping_r = {k: v for k, v in zip(range(1,101), rh_v)}

    # left
    k = np.array(list(mapping_l.keys()))
    v = np.array(list(mapping_l.values()))

    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
    mapping_ar[k] = v
    left_betas = mapping_ar[schaefer_l_lab]

    # right
    k = np.array(list(mapping_r.keys()))
    v = np.array(list(mapping_r.values()))

    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
    mapping_ar[k] = v
    right_betas = mapping_ar[schaefer_r_lab]

    vmax = np.max(axis_avg)
    vmin = np.min(axis_avg)

    if sym_cbar:
        bval = np.sort((np.abs(vmin), np.abs(vmax)))[1]
        vmax = bval
        vmin = -bval
    else:
        if vmin < 0:
            vmin = 0
        
    
    outd = f'figs/surface_plots/{model_name}'    
    if not os.path.exists(outd):
        os.makedirs(outd)

    # force threshold low to show full results, contour maps threshold
    nilearn_kwargs['threshold']=0.001

    reg = {}

    reg['left'] = np.argwhere(contour_bin_l == 1)+1
    reg['right'] = np.argwhere(contour_bin_r == 1)+1

    sch_atlas = SurfaceImage(mesh=fs_mapping['inflated'],
                                data={"left": schaefer_l_lab,
                                    "right": schaefer_r_lab})

    col = contour_col

    for hemi, betas in zip(['left', 'right'],[left_betas, right_betas]):
        regions_indices = reg[hemi]
        cols = [col for i in regions_indices]

        for view in ['medial', 'lateral']:
            # generate new figure- larger than default for saving
            fig, axes = plt.subplots(figsize=(10,10), subplot_kw={'projection': '3d'})
            plotting.plot_surf_stat_map(
                        fsaverage[f'infl_{hemi}'], betas, hemi=hemi,
                        colorbar=True, symmetric_cbar=sym_cbar, 
                        darkness=None, view=view, vmax=vmax, vmin=vmin,
                        figure=fig, axes=axes,
                        **nilearn_kwargs)
            plotting.plot_surf_contours(
                        roi_map=sch_atlas,
                        hemi=hemi,
                        view=view,
                        levels=regions_indices,
                        figure=fig, axes=axes,
                        colors=cols,
                        output_file=f'{outd}/{model_name}_{view}_{hemi}_contour_{mesh_qual}Q.png',
                    )
            
class ParcelwiseModel:
    # ensure parceldf and modeldf have 'SubID' column
    # ensure first row is 'Younger' AgeGroup

    def __init__(self, parceldf, modeldf, alpha=0.05):
        self.parceldf = parceldf
        self.parcels = parceldf.columns[1:]
        self.modeldf = modeldf
        self.mergedf = parceldf.merge(modeldf, on='subID')
        self.mergedf = self.mergedf.sort_values("AgeGroup", ascending=False)
        self.alpha = alpha

    def runModel(self, model_terms):

        self.model = " + ".join(model_terms)

        # handle statsmodels categorical labeling quirk
        self.alt_terms = []
        for term in model_terms:
            if 'C(' in term:
                G1 = self.mergedf[term.split('C(')[-1][:-1]][0]
                alt_term = f'{term}[T.{G1}]'
            else:
                alt_term = f'{term}'
            self.alt_terms.append(alt_term)

        # instantiate dict to save output
        self.modelterm_output = {f'{term}_{stat}':[] for term in self.alt_terms for stat in ('beta', 't', 'p')}
        self.modelterm_output['modelR2'] = []
        self.modelterm_output['modelF'] = []
       
        # run parcelwise model- save important output
        for parcel in self.parcels:
            res = smf.ols(f'{parcel} ~ {self.model}', data=self.mergedf).fit()

            for term in self.alt_terms:
                self.modelterm_output[f'{term}_beta'].append(res.params[term])
                self.modelterm_output[f'{term}_t'].append(res.tvalues[term])
                self.modelterm_output[f'{term}_p'].append(res.pvalues[term])
            self.modelterm_output['modelR2'].append(res.rsquared_adj)
            self.modelterm_output['modelF'].append(res.fvalue)
    
    def fdrCorrect(self):
        for term in self.alt_terms:
            _, corrected = fdrcorrection(
                self.modelterm_output[f'{term}_p'], alpha=self.alpha)
            self.modelterm_output[f'{term}_p_corrected'] = corrected
    
    def getSigParcels(self):

        self.sigparcels = {f'{term}':[] for term in self.alt_terms}
        for term in self.alt_terms:
            for p, parcel in zip(self.modelterm_output[f'{term}_p_corrected'], self.parcels):
                if p < self.alpha:
                    self.sigparcels[term].append(parcel)
    
    def reportParcels(self, terms='allsig'):


        filtered_term = {}
        filtered_term['parcel'] = []
        filtered_term['modelF'] = []
        filtered_term['modelR2'] = []
        for term in self.alt_terms:
                for stat in ('beta', 't', 'p_corrected'):
                    filtered_term[f'{term}_{stat}'] = []

        if terms == 'allsig':
            sig_noFD = {k: v for k, v in self.sigparcels.items() if k != "meanFD"}
            full_sigparcels = list(set([item for sublist in sig_noFD.values() for item in sublist]))      

            for parcel in full_sigparcels:
                parcelidx = np.where(self.parcels==parcel)[0][0]
                filtered_term['parcel'].append(parcel)
                filtered_term['modelR2'].append(self.modelterm_output['modelR2'][parcelidx])
                filtered_term['modelF'].append(self.modelterm_output['modelF'][parcelidx])
                for term in self.alt_terms:
                    filtered_term[f'{term}_beta'].append(self.modelterm_output[f'{term}_beta'][parcelidx])
                    filtered_term[f'{term}_t'].append(self.modelterm_output[f'{term}_t'][parcelidx])
                    filtered_term[f'{term}_p_corrected'].append(self.modelterm_output[f'{term}_p_corrected'][parcelidx])
        else:
            sig_noFD = {k: v for k, v in self.sigparcels.items() if k in terms}
            full_sigparcels = list(set([item for sublist in sig_noFD.values() for item in sublist]))  

            for parcel in full_sigparcels:
                parcelidx = np.where(self.parcels==parcel)[0][0]
                filtered_term['parcel'].append(parcel)
                filtered_term['modelR2'].append(self.modelterm_output['modelR2'][parcelidx])
                filtered_term['modelF'].append(self.modelterm_output['modelF'][parcelidx])
                for term in self.alt_terms:
                    filtered_term[f'{term}_beta'].append(self.modelterm_output[f'{term}_beta'][parcelidx])
                    filtered_term[f'{term}_t'].append(self.modelterm_output[f'{term}_t'][parcelidx])
                    filtered_term[f'{term}_p_corrected'].append(self.modelterm_output[f'{term}_p_corrected'][parcelidx])

        df = pd.DataFrame(filtered_term)
        df = df.round(decimals=3)

        return df
    
    def reportFull(self):
        filtered_term = {}
        filtered_term['parcel'] = []
        filtered_term['modelF'] = []
        filtered_term['modelR2'] = []
        for term in self.alt_terms:
            for stat in ('beta', 't', 'p_corrected'):
                filtered_term[f'{term}_{stat}'] = []
        
        for parcel in self.parcels:
            parcelidx = np.where(self.parcels==parcel)[0][0]
            filtered_term['parcel'].append(parcel)
            filtered_term['modelR2'].append(self.modelterm_output['modelR2'][parcelidx])
            filtered_term['modelF'].append(self.modelterm_output['modelF'][parcelidx])
            for term in self.alt_terms:
                filtered_term[f'{term}_beta'].append(self.modelterm_output[f'{term}_beta'][parcelidx])
                filtered_term[f'{term}_t'].append(self.modelterm_output[f'{term}_t'][parcelidx])
                filtered_term[f'{term}_p_corrected'].append(self.modelterm_output[f'{term}_p_corrected'][parcelidx])
        
        df = pd.DataFrame(filtered_term)
        df = df.round(decimals=3)

        return df
    
    def plotSurface(self, term, nilearn_kwargs, contour_col='r', contour_thresh=True, sym_cbar=False, mesh_qual='high'):

        axis_avg = self.modelterm_output[f'{term}_t'][:200]
        
        if contour_thresh:
            pval = self.modelterm_output[f'{term}_p_corrected'][:200]
            contour_bin = (pval < self.alpha).astype(int)
        else:
            contour_bin = np.ones(axis_avg.shape)

        plot_surf(axis_avg, contour_bin=contour_bin, 
                  model_name=term, contour_col = contour_col,
                  nilearn_kwargs = nilearn_kwargs, sym_cbar=sym_cbar, mesh_qual=mesh_qual)
  

#%%

# read in all beh docs
behavioral_data = pd.read_csv('./input_data/full_behavioral_data_public.csv')


#%% load in parcel ISC

# load in parcel isc data
parcel_isc = pd.read_csv('./input_data/full_roi_ISC_values.csv')

# rename first column to subID
parcel_isc = parcel_isc.rename(columns={parcel_isc.columns[0]: 'subID'})
# remove 17Networks_ from column names, replace - with _
parcel_isc.columns = [col.replace('17Networks_', '') for col in parcel_isc.columns]
parcel_isc.columns = [col.replace('-', '_') for col in parcel_isc.columns]
# fisher r-z transform
for parcel in parcel_isc.columns[1:]:
    parcel_isc[parcel] = np.arctanh(parcel_isc[parcel])


#%% H2A

# run behavioral awkwardness similarity model (H2A)
awk_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=behavioral_data)

init_model_terms = ['awk_sim_LOO', 'C(AgeGroup)', 'awk_sim_LOO:C(AgeGroup)', 'meanFD']
awk_model.runModel(model_terms = init_model_terms)

awk_model.fdrCorrect()
awk_model.getSigParcels()

awk_results_df = awk_model.reportParcels(terms=['awk_sim_LOO'])

contour_col = np.array((34, 255, 0))/255
awk_model.plotSurface('awk_sim_LOO', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')


#%% run second model with TOM (H2B)

tom_con_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=behavioral_data)

init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'NFY_control', 'meanFD']
tom_con_model.runModel(model_terms = init_model_terms)

tom_con_model.fdrCorrect()
tom_con_model.getSigParcels()

tom_results_con_df = tom_con_model.reportParcels(terms='NFY_mean')

tom_con_model.plotSurface('NFY_mean', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')

# correlation between output t statistics for both H2A and H2B models
r, p = stats.pearsonr(awk_model.modelterm_output['awk_sim_LOO_t'],
                      tom_con_model.modelterm_output['NFY_mean_t'])


#%% Supplemental analyses

tom_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=behavioral_data)

init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'meanFD']
tom_model.runModel(model_terms = init_model_terms)

tom_model.fdrCorrect()
tom_model.getSigParcels()

tom_results_df = tom_model.reportParcels(terms='NFY_mean')

tom_model.plotSurface('NFY_mean', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')

#%% get the full set of t stats from each of the 216 parcels 
r_sup, p_sup = stats.pearsonr(awk_model.modelterm_output['awk_sim_LOO_t'],
                              tom_model.modelterm_output['NFY_mean_t'])


#%% save output
if not os.path.exists('./output_results'):
    os.makedirs('./output_results')

awk_results_df.to_csv('./output_results/awk_results.csv', index=False)
tom_results_con_df.to_csv('./output_results/tom_con_results.csv', index=False)
tom_results_df.to_csv('./output_results/tom_results.csv', index=False)

#%% Behavioral replication analsyses

awkwardness_results = behavioral_data[['awk_sim_LOO', 'AgeGroup']].dropna()
tom_results = behavioral_data[['NFY_mean', 'AgeGroup']].dropna()

# run t-tests
awkwardness_ttest = stats.ttest_ind(awkwardness_results[awkwardness_results['AgeGroup'] == 'Younger']['awk_sim_LOO'],
                                    awkwardness_results[awkwardness_results['AgeGroup'] == 'Older']['awk_sim_LOO'],
                                    equal_var=False)

tom_ttest = stats.ttest_ind(tom_results[tom_results['AgeGroup'] == 'Younger']['NFY_mean'],
                            tom_results[tom_results['AgeGroup'] == 'Older']['NFY_mean'],
                            equal_var=False)

# %% print t-test results to only 2decimal

print("Awkwardness Similarity T-Test Results:")
print(f"t-statistic: {awkwardness_ttest.statistic:.2f}, p-value: {awkwardness_ttest.pvalue:.2f}")

print("\nTheory of Mind T-Test Results:")
print(f"t-statistic: {tom_ttest.statistic:.2f}, p-value: {tom_ttest.pvalue:.2f}")

# %% print means and standard deviations for the awk sim and tom  measures by age group

print("Awkwardness Similarity by Age Group:")
print(awkwardness_results.groupby('AgeGroup').agg(['mean', 'std']))

print("\nTheory of Mind by Age Group:")
print(tom_results.groupby('AgeGroup').agg(['mean', 'std']))

# %%
# run age t test on framewise displacement

fd_results = behavioral_data[['meanFD', 'AgeGroup']].dropna()

# run t-tests
fd_ttest = stats.ttest_ind(fd_results[fd_results['AgeGroup'] == 'Younger']['meanFD'],
                           fd_results[fd_results['AgeGroup'] == 'Older']['meanFD'],
                           equal_var=False)

print("\nmeanFD by Age Group:")
print(fd_results.groupby('AgeGroup').agg(['mean', 'std']))
print("\nmeanFD T-Test Results:")
print(f"t-statistic: {fd_ttest.statistic:.2f}, p-value: {fd_ttest.pvalue:.2f}")
# %%

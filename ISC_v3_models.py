#%%
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg

import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

import seaborn as sns


#%%
import os

from nilearn import surface
from nilearn import plotting
from nilearn import datasets
from nilearn.surface import SurfaceImage
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

#%%

def plot_surf(axis_avg, contour_bin, model_name, contour_col = 'g', nilearn_kwargs = {}, sym_cbar=False, mesh_qual='high'):

    lh_v = axis_avg[:100]
    rh_v = axis_avg[100:]
    contour_bin_l = contour_bin[:100]
    contour_bin_r = contour_bin[100:]

    # this needs to point to local copy of freesurfer schaefer parcellation
    bdir = '/home/roberto/Documents/Work/projects/ISC/data/freesurfer_parcs/FreeSurfer5.3'
    
    if mesh_qual.lower() == 'high':
        fs = 'fsaverage'
    elif mesh_qual.lower() == 'med':
        fs = 'fsaverage6'
    elif mesh_qual.lower() == 'low':
        fs = 'fsaverage5'
    else:
        raise ValueError('`mesh_qual` must be "high" or "low"')

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
            
#TODO create subcortex plotting - use glass brain
# def plot_subcort(axis_avg, contour_bin, model_name, contour_col = 'g'):
#     # generate nii object of subcortical tian parcellation
#     # load in tian parcellation
#     # load in tian labels
#     # iterate through labels- replace with axis_avg value
#     # use nilearn.plotting to plot
#     # iterate through contour_bin
#     # generate contour - how? need slice

#%% parcelwise analyses H2A

class ParcelwiseModel:
    # notes to self:
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
movement = pd.read_csv('~/Documents/Work/projects/ISC/data/full_sublist_movement-0.5_threshold.csv')
other_beh = pd.read_csv('~/Documents/Work/projects/ISC/data/forMRIanalysis_02-05-25.csv')

PC = pd.read_csv('~/Documents/Work/projects/ISC/data/full_sample_PC.csv', index_col=0)

awk_sim = pd.read_csv('~/Documents/Work/projects/ISC/data/awk_sim_score_LOO_version.csv')

#%%

beh_keep = [
    'age',
    'AgeGroup',
    'NFY_control',
    'NFY_emotion',
    'NFY_belief',
    'NFY_motivation',
    'NFY_fauxpas',
    'NFY_deception',
    'NFY_total',
    'awk_sim',
    'awk_sim_LOO',
    ]

move_keep = ['meanFD']
uds_keep = ['UDS_execfxn', 'UDS_epmem']
oa_uds_keep = ['UDS_OA_execfxn', 'UDS_OA_epmem']
ya_uds_keep = ['UDS_YA_execfxn', 'UDS_YA_epmem']

#%%

movement['subID'] = [f'sub-{int(i.split('-')[1])}' for i in movement['subj']]
other_beh['subID'] = [f'sub-{i}' for i in other_beh['IDnum']]
awk_sim['subID'] = [f'sub-{i}' for i in awk_sim['SUBID']]

# allsub
full_df = other_beh.merge(movement, on='subID', how='inner')
full_df = full_df.merge(PC, on='subID', how='inner')
full_df = full_df.merge(awk_sim, on='subID', how='left')


#trim
fullkeep = ['subID'] + beh_keep + move_keep + uds_keep + oa_uds_keep + ya_uds_keep + ['PC1']
full_df = full_df[fullkeep]
full_df['NFY_mean'] = full_df[['NFY_emotion', 'NFY_belief', 'NFY_motivation', 'NFY_fauxpas', 'NFY_deception']].mean(axis=1)


non_outlier_full_df = full_df[full_df['subID'] != 'sub-1321']

#%% generate plots first

plt.rcParams.update({'font.size': 12})

outlier_ver=False

if outlier_ver:
    plot_df = non_outlier_full_df
else:
    plot_df = full_df



#%% load in parcel ISC

# load in parcel isc data
parcel_isc = pd.read_csv('~/Documents/Work/projects/ISC/data/temp_move/full_roi_ISC_values.csv')

# rename first column to subID
parcel_isc = parcel_isc.rename(columns={parcel_isc.columns[0]: 'subID'})
# remove 17Networks_ from column names, replace - with _
parcel_isc.columns = [col.replace('17Networks_', '') for col in parcel_isc.columns]
parcel_isc.columns = [col.replace('-', '_') for col in parcel_isc.columns]
# fisher r-z transform
for parcel in parcel_isc.columns[1:]:
    parcel_isc[parcel] = np.arctanh(parcel_isc[parcel])



#%% find missing subjects between parcel_isc df and plot_df

missing_subs = set(parcel_isc['subID']) - set(plot_df['subID'])
print(f'Missing subjects: {missing_subs}') 

#%% H2A

# run orig parcel model
awk_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=plot_df)

init_model_terms = ['awk_sim_LOO', 'C(AgeGroup)', 'awk_sim_LOO:C(AgeGroup)', 'meanFD']
awk_model.runModel(model_terms = init_model_terms)

awk_model.fdrCorrect()
awk_model.getSigParcels()

awk_results_df = awk_model.reportParcels(terms=['awk_sim_LOO'])

#%% plot
#TODO need to find a way of displaying subcortical on the surface.
#TODO check if contour border can be made thicker- maybe only solution is to use lower quality mesh? maybe dind different contour color for clarity
#TODO create colormap with hot on positive and cool on negative, leave as symmetric cmap

{'cmap':'hot', 'threshold':0.0001}
contour_col = np.array((34, 255, 0))/255
# awk_model.plotSurface('awk_sim_LOO', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')


#%% run second model with TOM (full version) all parcels (once with nfy control and once without])

tom_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=plot_df)

init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'meanFD']
tom_model.runModel(model_terms = init_model_terms)

tom_model.fdrCorrect()
tom_model.getSigParcels()

tom_results_df = tom_model.reportParcels(terms='NFY_mean')




#%%

# tom_model.plotSurface('NFY_mean', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')

#%%

tom_con_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=plot_df)

init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'NFY_control', 'meanFD']
tom_con_model.runModel(model_terms = init_model_terms)

tom_con_model.fdrCorrect()
tom_con_model.getSigParcels()

tom_results_con_df = tom_con_model.reportParcels(terms='NFY_mean')

#%% get the full set of t stats from each of the 216 parcels 
r, p = stats.pearsonr(awk_model.modelterm_output['awk_sim_LOO_t'],
                      tom_con_model.modelterm_output['NFY_mean_t'])


#%% get the full set of t stats from each of the 216 parcels 
r_sup, p_sup = stats.pearsonr(awk_model.modelterm_output['awk_sim_LOO_t'],
                              tom_model.modelterm_output['NFY_mean_t'])

#%%

# tom_con_model.plotSurface('NFY_mean', {'threshold':0.0001}, contour_col=contour_col, sym_cbar=True, mesh_qual='high')

#%% save everything

awk_results_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/awk_results.csv', index=False)
tom_results_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/tom_results.csv', index=False)
tom_results_con_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/tom_con_results.csv', index=False)


#%%############################################################################################################################


# create a scatter of all subjects for all significant parcels
# pull sig parcels - filter df (.mergedf) - need only awk(or nfy mean), sig parcels (maybe long format to hue parcels? maybe messy), and age group!
# make a nice scatter plot - may be dealing with issues surrounding rasterization of all these points

# plt.gca().set_rasterization_zorder(1)
# plt.plot(randn(100),randn(100,500),"k",alpha=0.03,zorder=0)
# plt.savefig("test.pdf",dpi=90)

# decide on fit line after the fact- probably not, introduces stats that I would need to explain, unless the figure looks really bad. 
# this is just a visualization to help understand the pattern

awk_plot_df = awk_model.mergedf[awk_model.sigparcels['awk_sim_LOO']+['awk_sim_LOO', 'AgeGroup']]
tom_plot_df = tom_model.mergedf[tom_model.sigparcels['NFY_mean']+['NFY_mean', 'AgeGroup']]
tom_plot_con_df = tom_con_model.mergedf[tom_con_model.sigparcels['NFY_mean']+['NFY_mean', 'AgeGroup']]


#%%

awk_plot_df_melted = pd.melt(awk_plot_df,
                             id_vars=['AgeGroup', 'awk_sim_LOO'],
                             var_name='Parcel',
                             value_name='ISC_value')

tom_plot_df_melted = pd.melt(tom_plot_df,
                             id_vars=['AgeGroup', 'NFY_mean'],
                             var_name='Parcel',
                             value_name='ISC_value')

tom_plot_con_df_melted = pd.melt(tom_plot_con_df,
                             id_vars=['AgeGroup', 'NFY_mean'],
                             var_name='Parcel',
                             value_name='ISC_value')



#%%
# find overlap of sig parcels

awk_sig_parcels = set(awk_plot_df_melted['Parcel'])
tom_sig_parcels = set(tom_plot_df_melted['Parcel'])

overlap_parcels = awk_sig_parcels.intersection(tom_sig_parcels)

# find unique parcels for awk

awk_unique_parcels = awk_sig_parcels - tom_sig_parcels

#%%
# run linear models of the 5 rois for both awk and TOM

ment_roi = pd.read_csv('~/Documents/Work/projects/ISC/data/ment_roi_ISC_vals.csv', index_col=0)
ment_roi = ment_roi[list(ment_roi.columns[:7]) + ['subID']]
roi_relabel = ['dmPFC', 'rTPJ', 'l_temppole', 'precuneus', 'lTPJ', 'r_temppole', 'vmPFC']
roi_test = ['dmPFC', 'vmPFC', 'rTPJ', 'lTPJ', 'precuneus']
ment_roi.columns = roi_relabel + ['subID']

roi_df = ment_roi[['subID'] + roi_test]

for roi in roi_df.columns[1:]:
    roi_df[roi] = np.arctanh(roi_df[roi])

#%% run "parcelwise models" but only on the 5 rois
awk_roi_model = ParcelwiseModel(parceldf=roi_df, modeldf=plot_df)
init_model_terms = ['awk_sim_LOO', 'C(AgeGroup)', 'awk_sim_LOO:C(AgeGroup)', 'meanFD']
awk_roi_model.runModel(model_terms = init_model_terms)
awk_roi_model.fdrCorrect()
awk_roi_model.getSigParcels()

awk_roi_results_df = awk_roi_model.reportFull()


#%% do the same as above but for TOM

tom_roi_model = ParcelwiseModel(parceldf=roi_df, modeldf=plot_df)
init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'meanFD']
tom_roi_model.runModel(model_terms = init_model_terms)
tom_roi_model.fdrCorrect()
tom_roi_model.getSigParcels()

tom_roi_results_df = tom_roi_model.reportFull()


tom_roi_con_model = ParcelwiseModel(parceldf=roi_df, modeldf=plot_df)
init_model_terms = ['NFY_mean', 'C(AgeGroup)', 'NFY_mean:C(AgeGroup)', 'NFY_control', 'meanFD']
tom_roi_con_model.runModel(model_terms = init_model_terms)
tom_roi_con_model.fdrCorrect()
tom_roi_con_model.getSigParcels()

tom_roi_con_results_df = tom_roi_con_model.reportFull()

#%%
# cog_model = ParcelwiseModel(parceldf=parcel_isc, modeldf=plot_df)
# init_model_terms = ['UDS_execfxn', 'UDS_epmem', 'C(AgeGroup)', 'meanFD']
# cog_model.runModel(model_terms = init_model_terms)
# cog_model.fdrCorrect()
# cog_model.getSigParcels()

# cog_results_df = cog_model.reportParcels(terms=['UDS_execfxn', 'UDS_epmem'])

# %%

awk_roi_results_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/awk_roi_results.csv', index=False)
tom_roi_results_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/tom_roi_results.csv', index=False)
tom_roi_con_results_df.to_csv('~/Documents/Work/projects/ISC/data/1-5-26_results/tom_roi_con_results.csv', index=False)

# %%

# for the 5 ROIs, run regression to residualise the effects of meanFD from awk_sim_LOO,
# then plot a regplot of the residuals against the region similarity hue by AgeGroup

# use the merged dataframe from the ROI parcelwise model (contains awk_sim_LOO, meanFD, AgeGroup, and ROI values)
merged_roi = awk_roi_model.mergedf.copy()

# regress awk_sim_LOO on meanFD and keep residuals
fd_model = smf.ols('awk_sim_LOO ~ meanFD', data=merged_roi).fit()
merged_roi['awk_sim_LOO_resid'] = fd_model.resid


# create output directory for ROI regplots
outdir = 'figs/roi_regplots'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# define a pastel palette that switches colors between Older and Younger explicitly
# (maps 'Younger' -> light pastel orange, 'Older' -> light pastel blue). Falls back to a pastel palette if names differ.
ya_col = "#e7833f"  # light pastel orange
oa_col = "#3074db"    # light pastel blue

palette = {'Younger': ya_col, 'Older': oa_col}

# Generate a regplot per ROI. Use seaborn.lmplot which supports hue (AgeGroup) and per-hue regression lines.
for roi in roi_test:
    plot_df_roi = merged_roi[['awk_sim_LOO_resid', roi, 'AgeGroup']].dropna()

    # seaborn.lmplot creates its own figure; set aesthetics and save each plot
    g = sns.lmplot(x='awk_sim_LOO_resid', y=roi, hue='AgeGroup', data=plot_df_roi,
                   height=5, aspect=1.2, scatter_kws={'alpha':0.8, 's':30, 'edgecolor':'w'},
                   palette=palette)
    g.set_axis_labels('Beh. Similarity (residualised for meanFD)', f'{roi} ISC (Fisher z)')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Beh. Similarity vs {roi} ISC by AgeGroup')

    fname = f"{outdir}/{roi}_beh_similarity_resid_vs_{roi}_by_AgeGroup.svg"
    plt.savefig(fname)
    plt.show()

#%% same as above but for TOM, also residualize out NFY_con as well as meanFD

# regress awk_sim_LOO_resid on NFY_con and keep residuals
fd_model = smf.ols('NFY_mean ~ meanFD + NFY_control', data=merged_roi).fit()
merged_roi['NFY_mean_resid'] = fd_model.resid

# Generate a regplot per ROI. Use seaborn.lmplot which supports hue (AgeGroup) and per-hue regression lines.
for roi in roi_test:
    plot_df_roi = merged_roi[['NFY_mean_resid', roi, 'AgeGroup']].dropna()

    # seaborn.lmplot creates its own figure; set aesthetics and save each plot
    g = sns.lmplot(x='NFY_mean_resid', y=roi, hue='AgeGroup', data=plot_df_roi,
                   height=5, aspect=1.2, scatter_kws={'alpha':0.8, 's':30, 'edgecolor':'w'},
                   palette=palette)
    g.set_axis_labels('Theory of mind (residualised for meanFD and control Q.)', f'{roi} ISC (Fisher z)')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Theory of mind vs {roi} ISC by AgeGroup')

    fname = f"{outdir}/{roi}_Theory_of_mind_resid_vs_{roi}_by_AgeGroup.svg"
    plt.savefig(fname)

# %% run t test to look at age group differences in awkwardness similarity and theory of mind

awkwardness_results = full_df[['awk_sim_LOO', 'AgeGroup']].dropna()
tom_results = full_df[['NFY_mean', 'AgeGroup']].dropna()

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

fd_results = full_df[['meanFD', 'AgeGroup']].dropna()

# run t-tests
fd_ttest = stats.ttest_ind(fd_results[fd_results['AgeGroup'] == 'Younger']['meanFD'],
                            fd_results[fd_results['AgeGroup'] == 'Older']['meanFD'],
                            equal_var=False)


print("\nmeanFD by Age Group:")
print(fd_results.groupby('AgeGroup').agg(['mean', 'std']))
print("\nmeanFD T-Test Results:")
print(f"t-statistic: {fd_ttest.statistic:.2f}, p-value: {fd_ttest.pvalue:.2f}")
# %%

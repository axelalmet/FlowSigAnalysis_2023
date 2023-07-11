### Script to simulate the evolution of a ligand-receptor model. As a first step, we're considering just one ligand, one receptor,
### and one complex on a network.
from scipy.sparse import csr_matrix
import numpy as np
import seaborn as sns
import scanpy as sc
import pandas as pd
import networkx as nx

# Set a bunch of parameters that we need
end_time = 10.0
num_realisations = 5

# Generate the grid
num_rows = 10
num_cols = 90
G = nx.grid_2d_graph(num_cols, num_rows)
A = nx.adjacency_matrix(G).toarray() # Adjacency

### Set-up for analysis
simulation_path = '../output/'
    
wt_sols = []
pert_sols = []
for seed in range(num_realisations):

    wt_sol = pd.read_csv(simulation_path + "/SHH_BMP4_DORSOVENTRAL/shh_bmp4_dorsoventral_grn_network_ode_sol_" + str(seed) + ".csv")
    wt_sol_final = wt_sol[wt_sol['t'] == wt_sol['t'].max()]
    
    adata_cont = sc.AnnData(X=wt_sol_final[['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V']].values,
                            var=pd.DataFrame(index=['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V'],
                                             data={'type':['ligand', 'receptor', 'bound_complex', 'ligand', 'receptor', 'bound_complex', 'factor', 'factor', 'factor']}))
    adata_cont.obs['Condition'] = 'WT'
    adata_cont.obs['Sample'] = 'WT_' + str(seed + 1)
    
    adata_cont.obs['Condition'] = pd.Series(adata_cont.obs['Condition'].values, dtype='category').values
    adata_cont.obs['Sample'] = pd.Series(adata_cont.obs['Sample'].values, dtype='category').values
    adata_cont.obsm['spatial'] = wt_sol_final[['x', 'y']].values
    adata_cont.var_names = pd.Index(['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V'])

    # Construct the leiden blocks for potential sampling
    adata_cont.obsp['spatial_connectivities'] = csr_matrix(A)
    sc.tl.leiden(adata_cont, resolution=0.05, key_added='leiden_block', adjacency=csr_matrix(A))
    adata_cont.obs['leiden_block'] = pd.Series(adata_cont.obs['leiden_block'].values, dtype='string').values \
                                    + '_' \
                                    + pd.Series(adata_cont.obs['Sample'].values, dtype='string').values
    adata_cont.obs['leiden_block'] = pd.Series(adata_cont.obs['leiden_block'].values, dtype='category').values

    pert_sol = pd.read_csv(simulation_path + "/SHH_BMP4_DORSOVENTRAL/shh_bmp4_dorsoventral_grn_network_ode_sol_pert_" + str(seed) + ".csv")
    pert_sol_final = pert_sol[pert_sol['t'] == pert_sol['t'].max()]

    adata_pert = sc.AnnData(X=pert_sol_final[['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V']].values,
                            var=pd.DataFrame(index=['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V'],
                                             data={'type':['ligand', 'receptor', 'bound_complex', 'ligand', 'receptor', 'bound_complex', 'factor', 'factor', 'factor']}))    
    adata_pert.obs['Condition'] = 'PERT'
    adata_pert.obs['Sample'] = 'PERT_' + str(seed + 1)
    adata_pert.obs['Condition'] = pd.Series(adata_pert.obs['Condition'].values, dtype='category').values
    adata_pert.obs['Sample'] = pd.Series(adata_pert.obs['Sample'].values, dtype='category').values
    adata_pert.obsm['spatial'] = pert_sol_final[['x', 'y']].values
    adata_pert.var_names = pd.Index(['Shh', 'Ptch1', 'Shh_bound', 'Bmp4', 'Bmpr1A_Bmpr2', 'Bmp4_bound', 'D', 'I', 'V'])

    adata_pert.obsp['spatial_connectivities'] = csr_matrix(A)
    sc.tl.leiden(adata_pert, resolution=0.05, key_added='leiden_block', adjacency=csr_matrix(A))
    adata_pert.obs['leiden_block'] = pd.Series(adata_pert.obs['leiden_block'].values, dtype='string').values \
                                    + '_' \
                                    + pd.Series(adata_pert.obs['Sample'].values, dtype='string').values
    adata_pert.obs['leiden_block'] = pd.Series(adata_pert.obs['leiden_block'].values, dtype='category').values

    wt_sols.append(adata_cont)
    pert_sols.append(adata_pert)

adata_wt = wt_sols[0].concatenate(wt_sols[1:])
adata_pert = pert_sols[0].concatenate(pert_sols[1:])

adata = adata_wt.concatenate(adata_pert)

# Construct the neighborhood graph
spatial_graph = np.zeros((adata.n_obs, adata.n_obs))

for sample in adata.obs['Sample'].unique():
    sample_indices = np.where(adata.obs['Sample'] == sample)[0]
    spatial_graph[np.ix_(sample_indices, sample_indices)] = A

adata.obsp['spatial_connectivities'] = csr_matrix(spatial_graph)

adata.layers['sols'] = adata.X.copy()

sc.pp.filter_cells(adata, min_genes=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Calculate the sp
print(adata.obs['leiden_block'])

adata.write(simulation_path + "/SHH_BMP4_DORSOVENTRAL/adata_shh_bmp4_DIV.h5ad", compression='gzip')

import os
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import commot as ct

# Import data
data_directory = '../data/'

# Load the scanpy object 
adata_chen = sc.read(data_directory + 'Mouse_embryo_E9.5.h5ad')

# Load the cellchat dataframe
df_cellchat = ct.pp.ligand_receptor_database(species='mouse', signaling_type='Secreted Signaling')

# Filter the cellchat dataframe
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata_chen, min_cell_pct=0.05)

# Run COMMOT
ct.tl.spatial_communication(adata_chen,
                            database_name='cellchat',
                            df_ligrec=df_cellchat_filtered,
                            dis_thr=20,
                            heteromeric=True,
                            pathway_sum=True)

adata_chen.write(data_directory + "Mouse_embryo_E9.5.h5ad", compression=True)


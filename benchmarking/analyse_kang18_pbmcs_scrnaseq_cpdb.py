import scanpy as sc
import pandas as pd
import flowsig as fs

adata = sc.read('../data/kang18_counts_25k.h5ad')

condition_key = 'condition'

adata.layers['normalized'] = adata.X.copy()

# We construct 20 gene expression modules using the raw cell count.
fs.pp.construct_gems_using_pyliger(adata,
                                n_gems = 20,
                                layer_key = 'counts',
                                condition_key = condition_key)

adata.X = adata.layers['normalized']
# Make sure your keys for these align with their condition labels
cpdb_results = {}
cpdb_active_tfs = {}
cpdb_results['ctrl'] = pd.read_csv('../communication_inference/output/kang18_counts_25k_ctrl_cpdb_method3/degs_analysis_significant_means_02_05_2024_235611.txt', sep='\t')
cpdb_active_tfs['ctrl'] = pd.DataFrame()

cpdb_results['stim'] = pd.read_csv('../communication_inference/output/kang18_counts_25k_stim_cpdb_method3/degs_analysis_significant_means_02_05_2024_235627.txt', sep='\t')
cpdb_active_tfs['stim'] = pd.read_csv('../communication_inference/output/kang18_counts_25k_stim_cpdb_method3/degs_analysis_CellSign_active_interactions_deconvoluted_02_05_2024_235627.txt', sep='\t')

# Subset for only secreted ligand-receptor interactions
for cond in cpdb_results:
    sig_means = cpdb_results[cond]
    cpdb_results[cond] = sig_means[(sig_means['secreted'] == True)&(sig_means['directionality'] == 'Ligand-Receptor')]
    
cellphonedb_output_key = 'cellphonedb_output'
adata.uns[cellphonedb_output_key] = cpdb_results
cellphonedb_tfs_key = 'cellphonedb_active_tfs'
adata.uns[cellphonedb_tfs_key] = cpdb_active_tfs

# We first construct the potential cellular flows from the cellchat output
fs.pp.construct_flows_from_cellphonedb(adata,
                                cellphonedb_output_key,
                                cellphonedb_tfs_key,
                                gem_expr_key = 'X_gem',
                                scale_gem_expr = True,
                                model_organism = 'human',
                                flowsig_network_key = 'flowsig_network_cpdb',
                                flowsig_expr_key = 'X_flow_cpdb')

# Delete teh cellphonedb output otherwise there will be trouble when saving the anndata objct
del adata.uns[cellphonedb_output_key]
del adata.uns[cellphonedb_tfs_key]

# Then we subset for "differentially flowing" variables
fs.pp.determine_informative_variables(adata,  
                                    flowsig_expr_key = 'X_flow_cpdb',
                                    flowsig_network_key = 'flowsig_network_cpdb',
                                    spatial = False,
                                    condition_key = condition_key,
                                    control_key =  'ctrl',
                                    qval_threshold = 0.05,
                                    logfc_threshold = 0.5)

# Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        condition_key = condition_key,
                        control_key = 'ctrl', 
                        flowsig_key = 'flowsig_network_cpdb',
                        flow_expr_key = 'X_flow_cpdb',
                        use_spatial = False,
                        n_jobs = 4,
                        n_bootstraps = 500)

# Now we do post-learning validation to reorient the network and remove low-quality edges.
# This part is key for reducing false positives
fs.tl.apply_biological_flow(adata,
                        flowsig_network_key = 'flowsig_network_cpdb',
                        adjacency_key = 'adjacency',
                        validated_key = 'validated')

edge_threshold = 0.7

fs.tl.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network_cpdb',
                                adjacency_key = 'adjacency_validated',
                                filtered_key = 'filtered')

adata.write('../data/kang18_counts_25k.h5ad', compression='gzip')

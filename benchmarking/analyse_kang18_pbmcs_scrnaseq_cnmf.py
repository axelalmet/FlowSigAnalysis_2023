import scanpy as sc
import pandas as pd
import flowsig as fs

adata = sc.read('../data/kang18_counts_25k.h5ad')

condition_key = 'condition'


# Make sure your keys for these align with their condition labels
cellchat_ctrl = pd.read_csv('../communication_inference/output/kang18_counts_25k_communications_ctrl.csv')
cellchat_stim = pd.read_csv('../communication_inference/output/kang18_counts_25k_communications_stim.csv')

cellchat_output_key = 'cellchat_output'
adata.uns[cellchat_output_key] = {'ctrl': cellchat_ctrl,
                                  'stim': cellchat_stim}

## All cNMF info has been stored in adata_kang.uns['cnmf_info']
##  and the cellwise memberships have been scored as adata_kang.obsm['X_gem_cnmf']

# We first construct the potential cellular flows from the cellchat output
fs.pp.construct_flows_from_cellchat(adata,
                                cellchat_output_key,
                                gem_expr_key = 'X_gem_cnmf',
                                scale_gem_expr = True,
                                model_organism = 'human',
                                flowsig_network_key = 'flowsig_network_cnmf',
                                flowsig_expr_key = 'X_flow_cnmf')

# Then we subset for "differentially flowing" variables
fs.pp.determine_informative_variables(adata,  
                                    flowsig_expr_key = 'X_flow_cnmf',
                                    flowsig_network_key = 'flowsig_network_cnmf',
                                    spatial = False,
                                    condition_key = condition_key,
                                    control_key =  'ctrl',
                                    qval_threshold = 0.05,
                                    logfc_threshold = 0.5)

# Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        condition_key = condition_key,
                        control_key = 'ctrl', 
                        flowsig_key = 'flowsig_network_cnmf',
                        flow_expr_key = 'X_flow_cnmf',
                        use_spatial = False,
                        n_jobs = 4,
                        n_bootstraps = 500)

# Now we do post-learning validation to reorient the network and remove low-quality edges.
# This part is key for reducing false positives
fs.tl.apply_biological_flow(adata,
                        flowsig_network_key = 'flowsig_network_cnmf',
                        adjacency_key = 'adjacency',
                        validated_key = 'validated')

edge_threshold = 0.7

fs.tl.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network_cnmf',
                                adjacency_key = 'adjacency_validated',
                                filtered_key = 'filtered')

adata.write('../data/kang18_counts_25k.h5ad', compression='gzip')

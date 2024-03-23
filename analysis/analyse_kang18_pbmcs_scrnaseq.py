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
cellchat_ctrl = pd.read_csv('../communication_inference/output/kang18_counts_25k_communications_ctrl.csv')
cellchat_stim = pd.read_csv('../communication_inference/output/kang18_counts_25k_communications_stim.csv')

cellchat_output_key = 'cellchat_output'
adata.uns[cellchat_output_key] = {'ctrl': cellchat_ctrl,
                                  'stim': cellchat_stim}

# # We first construct the potential cellular flows from the cellchat output
fs.pp.construct_flows_from_cellchat(adata,
                                cellchat_output_key,
                                gem_expr_key = 'X_gem',
                                scale_gem_expr = True,
                                model_organism = 'human',
                                flowsig_network_key = 'flowsig_network',
                                flowsig_expr_key = 'X_flow')

# Then we subset for "differentially flowing" variables
fs.pp.determine_informative_variables(adata,  
                                    flowsig_expr_key = 'X_flow',
                                    flowsig_network_key = 'flowsig_network',
                                    spatial = False,
                                    condition_key = condition_key,
                                    control_key =  'ctrl',
                                    qval_threshold = 0.05,
                                    logfc_threshold = 0.5)

adata.write(data_directory + 'kang18_counts_25k.h5ad', compression='gzip')

# Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        condition_key = condition_key,
                        control_key = 'ctrl', 
                        flowsig_key = 'flowsig_network',
                        flow_expr_key = 'X_flow',
                        use_spatial = False,
                        n_jobs = 4,
                        n_bootstraps = 500)

# Now we do post-learning validation to reorient the network and remove low-quality edges.
# This part is key for reducing false positives
fs.tl.apply_biological_flow(adata,
                        flowsig_network_key = 'flowsig_network',
                        adjacency_key = 'adjacency',
                        validated_key = 'validated')

edge_threshold = 0.7

fs.tl.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network',
                                adjacency_key = 'adjacency_validated',
                                filtered_key = 'filtered')

adata.write('../data/kang18_counts_25k.h5ad', compression='gzip')

import scanpy as sc
import pandas as pd
import flowsig as fs

adata = sc.read('../data/brain_organoid_D18_D35.h5ad')
condition_key = 'day'

# We construct 10 gene expression modules using the raw cell count.
fs.construct_gems_using_pyliger(adata,
                                n_gems = 20,
                                layer_key = 'counts',
                                condition_key = condition_key)

# Make sure your keys for these align with their condition labels
cellchat_D18 = pd.read('../communication_inference/output/D18_communications.csv')
cellchat_D35 = pd.read('../communication_inference/output/D35_communications.csv')

cellchat_output_key = 'cellchat_output'
adata.uns[cellchat_output_key] = {'D18': cellchat_D18,
                                  'D35': cellchat_D35}

# We first construct the potential cellular flows from the cellchat output
fs.construct_flows_from_cellchat(adata,
                                cellchat_output_key,
                                gem_expr_key = 'X_gem',
                                scale_gem_expr = True,
                                model_organism = 'human',
                                flowsig_network_key = 'flowsig_network',
                                flowsig_expr_key = 'X_flow')

# Then we subset for "differentially flowing" variables
fs.determine_informative_variables(adata,  
                                    flowsig_expr_key = 'X_flow',
                                    flowsig_network_key = 'flowsig_network',
                                    spatial = False,
                                    condition_key = condition_key,
                                    qval_threshold = 0.05,
                                    logfc_threshold = 0.5)

# Now we are ready to learn the network
fs.learn_intercellular_flows(adata,
                        condition_key = condition_key,
                        control_key = 'D18', 
                        flowsig_key = 'flowsig_network',
                        flow_expr_key = 'X_flow',
                        use_spatial = False,
                        n_jobs = 4,
                        n_bootstraps = 500)

# Now we do post-learning validation to reorient the network and remove low-quality edges.
# This part is key for reducing false positives
fs.apply_biological_flow(adata,
                        flowsig_network_key = 'flowsig_network',
                        adjacency_key = 'adjacency',
                        validated_adjacency_key = 'adjacency_validated')

edge_threshold = 0.7

fs.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network',
                                adjacency_key = 'adjacency',
                                filtered_adjacency_key = 'adjacency_filtered')

adata.write('../data/brain_organoid_D18_D35.h5ad', compression='gzip')
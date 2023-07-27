import scanpy as sc
import pandas as pd
import flowsig as fs

adata = sc.read('../data/liao20_sub.h5ad')
condition_key = 'group'

# We construct 10 gene expression modules using the raw cell count.
# fs.pp.construct_gems_using_pyliger(adata,
#                                 n_gems = 20,
#                                 layer_key = 'counts',
#                                 condition_key = condition_key)

# Make sure your keys for these align with their condition labels
# cellchat_HC = pd.read_csv('../communication_inference/output/liao20_sub_communications_HC.csv')
# cellchat_M = pd.read_csv('../communication_inference/output/liao20_sub_communications_M.csv')
# cellchat_S = pd.read_csv('../communication_inference/output/liao20_sub_communications_S.csv')

# cellchat_output_key = 'cellchat_output'
# adata.uns[cellchat_output_key] = {'HC': cellchat_HC,
#                                   'M': cellchat_M,
#                                   'S': cellchat_S}

# We first construct the potential cellular flows from the cellchat output
# fs.pp.construct_flows_from_cellchat(adata,
#                                 cellchat_output_key,
#                                 gem_expr_key = 'X_gem',
#                                 scale_gem_expr = True,
#                                 model_organism = 'human',
#                                 flowsig_network_key = 'flowsig_network',
#                                 flowsig_expr_key = 'X_flow')

# # Then we subset for "differentially flowing" variables
# fs.pp.determine_informative_variables(adata,  
#                                     flowsig_expr_key = 'X_flow',
#                                     flowsig_network_key = 'flowsig_network',
#                                     spatial = False,
#                                     condition_key = condition_key,
#                                     control_key =  'HC',
#                                     qval_threshold = 0.05,
#                                     logfc_threshold = 0.5)

# Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        condition_key = condition_key,
                        control_key = 'HC', 
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

edge_threshold = 0.8

fs.tl.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network',
                                adjacency_key = 'adjacency',
                                filtered_key = 'filtered')

adata.write('../data/liao20_sub.h5ad', compression='gzip')
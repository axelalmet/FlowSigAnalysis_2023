import scanpy as sc
import pandas as pd
import flowsig as fs

adata = sc.read('../data/chen22_E9.5_svg.h5ad')

# We construct 10 gene expression modules using the raw cell count.
# fs.pp.construct_gems_using_nsf(adata,
#                             n_gems = 20,
#                             layer_key = 'count',
#                             n_inducing_pts = 500,
#                             length_scale = 10)

commot_output_key = 'commot-cellchat'

# We first construct the potential cellular flows from the commot output
fs.pp.construct_flows_from_commot(adata,
                                commot_output_key,
                                gem_expr_key = 'X_gem',
                                scale_gem_expr = True,
                                flowsig_network_key = 'flowsig_network',
                                flowsig_expr_key = 'X_flow')

print(adata.uns['flowsig_network']['flow_var_info'])
    
# Then we subset for "spatially flowing" inflows and outflows
fs.pp.determine_informative_variables(adata,  
                                    flowsig_expr_key = 'X_flow',
                                    flowsig_network_key = 'flowsig_network',
                                    spatial = True,
                                    moran_threshold = 0.15,
                                    coord_type = 'grid',
                                    n_neighbours = 8,
                                    library_key = None)


print(adata.uns['flowsig_network']['flow_var_info'].index.tolist())

# # Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        flowsig_key = 'flowsig_network',
                        flow_expr_key = 'X_flow',
                        use_spatial = True,
                        block_key = 'spatial_kmeans',
                        n_jobs = 4,
                        n_bootstraps = 500)

# # Now we do post-learning validation to reorient the network and remove low-quality edges.
# # This part is key for reducing false positives
# fs.tl.apply_biological_flow(adata,
#                         flowsig_network_key = 'flowsig_network',
#                         adjacency_key = 'adjacency',
#                         validated_adjacency_key = 'adjacency_validated')

# edge_threshold = 0.85

# fs.tl.filter_low_confidence_edges(adata,
#                                 edge_threshold = edge_threshold,
#                                 flowsig_network_key = 'flowsig_network',
#                                 adjacency_key = 'adjacency',
#                                 filtered_adjacency_key = 'adjacency_filtered')

adata.write('../data/chen22_E9.5_svg.h5ad', compression='gzip')
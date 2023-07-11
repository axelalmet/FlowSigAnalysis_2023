import scanpy as sc
import pandas as pd
import numpy as np
from typing import List, Tuple
import networkx as nx
import random as rm
from causaldag import unknown_target_igsp, gsp
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
from sklearn.utils import safe_mask
from timeit import default_timer as timer
from joblib import Parallel, delayed
from pathlib import Path
import anndata as ad
from functools import reduce

def run_gsp(adata: ad.AnnData,
            init_perms: bool = False,
            alpha_ci: float = 1e-3,
            n_init_perms: int = 20,
            seed: int = 0):

    # Reseed the random number generator
    np.random.seed(seed) # Set the seed for reproducibility reasons

    # Get the number of samples for each dataframe  
    genes = adata.var_names

    samples = adata.X # Define the control data 

    num_samples = samples.shape[0]

    # Subsample with replacement
    subsamples = np.random.choice(num_samples, num_samples)
    
    resampled = samples[safe_mask(samples, subsamples), :]

    # We need to subset the gene expression matrices for genes with non-zero standard deviation in BOTH cases
    resampled_std = resampled.std(0)

    nonzero_gene_indices = resampled_std.nonzero()[0]

    # Re-subsample the data for non-zero STD variables
    resampled = resampled[:, nonzero_gene_indices]

    # Subset based on the genes with zero std in both cases
    considered_genes = list([genes[ind] for ind in nonzero_gene_indices])
    
    nodes = set(considered_genes)

    ### Run UT-IGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(resampled, invert=True)

    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha_ci)

    if init_perms:

        # Construct random permutations
        inflow_signals = [gene for gene in adata[:, adata.var['type'] == 'bound_complex'].var_names]
        factors = [gene for gene in adata[:, adata.var['type'] == 'factor'].var_names]
        outflow_signals = [gene for gene in adata[:, adata.var['type'] == 'ligand'].var_names]
        
        nodes = set(considered_genes)

        # Convert the permutations in terms of node indices
        converted_perms = []

        for iter in range(n_init_perms):

            converted_perm = []

            # Add inflowing signals
            rm.shuffle(inflow_signals)
            for node in inflow_signals:
                converted_perm.append(considered_genes.index(node))

            # Add GEMs
            rm.shuffle(factors)
            for node in factors:
                converted_perm.append(considered_genes.index(node))

            # Add outflowing signals
            rm.shuffle(outflow_signals)
            for node in outflow_signals:
                converted_perm.append(considered_genes.index(node))

            converted_perms.append(converted_perm)

        est_dag = gsp(nodes,
                    ci_tester,
                    nruns=len(converted_perms),
                    initial_permutations=converted_perms)
    
        # Convert to interventional CPDAG, i.e. I-MEC
        est_cpdag = est_dag.cpdag()
        adjacency_cpdag = est_cpdag.to_amat()[0]
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag}

    else:
        est_dag = gsp(nodes,
                    ci_tester,
                    nruns=20)
        
        # Convert to interventional CPDAG, i.e. I-MEC
        est_cpdag = est_dag.cpdag()
        adjacency_cpdag = est_cpdag.to_amat()[0]
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag}

def run_gsp_spatial(adata: ad.AnnData,
            block_key: str = 'leiden_block',
            init_perms: bool = False,
            alpha_ci: float = 1e-3,
            n_init_perms: int = 20,
            seed: int = 0):

    # Reseed the random number generator
    np.random.seed(seed) # Set the seed for reproducibility reasons

    # Get the number of samples for each dataframe  
    num_samples = adata.n_obs
    genes = adata.var_names
    
    # Get the gene expression matrix
    samples = adata.X

    # Define the blocks for bootstrapping
    block_clusters = sorted(adata.obs[block_key].unique().tolist())
    
    # Subsample WITHIN blocks replacement
    resampled = samples.copy()
    for block in block_clusters:
        block_indices = np.where(adata.obs[block_key] == block)[0] # Sample only those cells within the block
        
        block_subsamples = np.random.choice(block_indices, len(block_indices))
        resampled[block_indices, :] = samples[safe_mask(samples, block_subsamples), :]

    # We need to subset the gene expression matrices for genes with non-zero standard deviation in BOTH cases
    resampled_std = resampled.std(0)

    nonzero_gene_indices = resampled_std.nonzero()[0]

    # Subset based on the genes with zero std in both cases
    considered_genes = list([genes[ind] for ind in nonzero_gene_indices])
    
    nodes = set(considered_genes)

    resampled = resampled[:, nonzero_gene_indices]

    ### Run UT-IGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(resampled, invert=True)

    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha_ci)

    if init_perms:

        # Construct random permutations
        
        inflow_signals = [gene for gene in adata[:, adata.var['type'] == 'bound_complex'].var_names]
        factors = [gene for gene in adata[:, adata.var['type'] == 'factor'].var_names]
        outflow_signals = [gene for gene in adata[:, adata.var['type'] == 'ligand'].var_names]
        
        nodes = set(considered_genes)

        # Convert the permutations in terms of node indices
        converted_perms = []

        for iter in range(n_init_perms):

            converted_perm = []

            # Add inflowing signals
            rm.shuffle(inflow_signals)
            for node in inflow_signals:
                converted_perm.append(considered_genes.index(node))

            # Add GEMs
            rm.shuffle(factors)
            for node in factors:
                converted_perm.append(considered_genes.index(node))

            # Add outflowing signals
            rm.shuffle(outflow_signals)
            for node in outflow_signals:
                converted_perm.append(considered_genes.index(node))

            converted_perms.append(converted_perm)

        est_dag = gsp(nodes,
                    ci_tester,
                    nruns=len(converted_perms),
                    initial_permutations=converted_perms)
    
        # Convert to interventional CPDAG, i.e. I-MEC
        est_cpdag = est_dag.cpdag()
        adjacency_cpdag = est_cpdag.to_amat()[0]
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag}

    else:
        est_dag = gsp(nodes,
                    ci_tester,
                    nruns=20)
        
        # Convert to interventional CPDAG, i.e. I-MEC
        est_cpdag = est_dag.cpdag()
        adjacency_cpdag = est_cpdag.to_amat()[0]
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag}

def run_utigsp(adata: ad.AnnData,
                condition_key: str = 'Condition',
                control_key: str = 'WT',
                perturbed_keys: List[str] = ['PERT'],
                init_perms: bool = False,
                n_init_perms: int = 20,
                alpha_ci: float = 1e-3,
                alpha_inv: float = 1e-3,
                seed: int = 0):

    # Reseed the random number generator
    np.random.seed(seed) # Set the seed for reproducibility reasons

    # Get the number of samples for each dataframe  
    genes = adata.var_names

    adata_control = adata[adata.obs[condition_key] == control_key]
    control_samples = adata_control.X # Define the control data
    
    adata_perturbed = adata[adata.obs[condition_key] != control_key]
    perturbed_samples = [adata_perturbed[adata_perturbed.obs[condition_key] == cond].X  for cond in perturbed_keys] # Get the perturbed data
        # Resample with replacement for bootstrapping
    perturbed_resampled = []    

    # Just sub-sample across all cells per condition
    num_samples_control = control_samples.shape[0]
    num_samples_perturbed = [sample.shape[0] for sample in perturbed_samples]

    # Subsample with replacement
    subsamples_control = np.random.choice(num_samples_control, num_samples_control)
    subsamples_perturbed = [np.random.choice(num_samples, num_samples) for num_samples in num_samples_perturbed]
    
    control_resampled = control_samples[safe_mask(control_samples, subsamples_control), :]

    for i in range(len(perturbed_samples)):
        num_subsamples = subsamples_perturbed[i]
        perturbed_sample = perturbed_samples[i]

        resampled = perturbed_sample[safe_mask(num_subsamples, num_subsamples), :]
        perturbed_resampled.append(resampled)

    # We need to subset the gene expression matrices for genes with non-zero standard deviation in BOTH cases
    control_resampled_std = control_resampled.std(0)
    perturbed_resampled_std = [sample.std(0) for sample in perturbed_resampled]

    nonzero_gene_indices_control = control_resampled_std.nonzero()[0]
    nonzero_gene_indices_perturbed = [resampled_std.nonzero()[0] for resampled_std in perturbed_resampled_std]

    nonzero_gene_indices = reduce(np.intersect1d, (nonzero_gene_indices_control, *nonzero_gene_indices_perturbed))

    # Re-subsample the data for non-zero STD variables
    control_resampled = control_resampled[:, nonzero_gene_indices]

    for i, resampled in enumerate(perturbed_resampled):
        perturbed_resampled[i] = resampled[:, nonzero_gene_indices]

    # Subset based on the genes with zero std in both cases
    considered_genes = list([genes[ind] for ind in nonzero_gene_indices])
    
    nodes = set(considered_genes)

    ### Run UT-IGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(control_resampled, invert=True)
    invariance_suffstat = gauss_invariance_suffstat(control_resampled, perturbed_resampled)

    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    ## Run UT-IGSP by considering all possible initial permutations
    setting_list = [dict(known_interventions=[]) for _ in perturbed_resampled]

    if init_perms:

        # Construct random permutations
        inflow_signals = [gene for gene in adata[:, adata.var['type'] == 'bound_complex'].var_names]
        factors = [gene for gene in adata[:, adata.var['type'] == 'factor'].var_names]
        outflow_signals = [gene for gene in adata[:, adata.var['type'] == 'ligand'].var_names]
        
        nodes = set(considered_genes)

        # Convert the permutations in terms of node indices
        converted_perms = []

        for iter in range(n_init_perms):

            converted_perm = []

            # Add inflowing signals
            rm.shuffle(inflow_signals)
            for node in inflow_signals:
                converted_perm.append(considered_genes.index(node))

            # Add GEMs
            rm.shuffle(factors)
            for node in factors:
                converted_perm.append(considered_genes.index(node))

            # Add outflowing signals
            rm.shuffle(outflow_signals)
            for node in outflow_signals:
                converted_perm.append(considered_genes.index(node))

            converted_perms.append(converted_perm)

        est_dag, est_targets_list = unknown_target_igsp(setting_list,
                                                        nodes,
                                                        ci_tester,
                                                        invariance_tester,
                                                        nruns=len(converted_perms),
                                                        initial_permutations=converted_perms)

        # Convert to interventional CPDAG, i.e. I-MEC
        est_icpdag = est_dag.interventional_cpdag(est_targets_list, cpdag=est_dag.cpdag())

        adjacency_cpdag = est_icpdag.to_amat()[0]

        perturbed_targets_list = []

        for i in range(len(est_targets_list)):
            targets_list = list(est_targets_list[i])
            targets_ligand_indices = [list(genes).index(considered_genes[target]) for target in targets_list]
            perturbed_targets_list.append(targets_ligand_indices)
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag,
                'perturbed_targets_indices':perturbed_targets_list}
    else:   

        est_dag, est_targets_list = unknown_target_igsp(setting_list,
                                                        nodes,
                                                        ci_tester,
                                                        invariance_tester,
                                                        nruns=20)

        # Convert to interventional CPDAG, i.e. I-MEC
        est_icpdag = est_dag.interventional_cpdag(est_targets_list, cpdag=est_dag.cpdag())

        adjacency_cpdag = est_icpdag.to_amat()[0]

        perturbed_targets_list = []

        for i in range(len(est_targets_list)):
            targets_list = list(est_targets_list[i])
            targets_ligand_indices = [list(genes).index(considered_genes[target]) for target in targets_list]
            perturbed_targets_list.append(targets_ligand_indices)
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag,
                'perturbed_targets_indices':perturbed_targets_list}

def run_utigsp_spatial(adata: ad.AnnData,
                block_key: str = 'leiden_block',
                condition_key: str = 'condition',
                control_key: str = 'WT',
                perturbed_keys: List[str] = ['PERT'],
                init_perms: bool = False,
                n_init_perms: int = 20,
                alpha_ci: float = 1e-3,
                alpha_inv: float = 1e-3,
                seed: int = 0):

    # Reseed the random number generator
    np.random.seed(seed) # Set the seed for reproducibility reasons

    # Get the number of samples for each dataframe  
    genes = adata.var_names

    adata_control = adata[adata.obs[condition_key] == control_key]
    control_samples = adata_control.X # Define the control data
    
    adata_perturbed = adata[adata.obs[condition_key] != control_key]
    perturbed_samples = [adata_perturbed[adata_perturbed.obs[condition_key] == cond].X  for cond in perturbed_keys] # Get the perturbed data
    perturbed_resampled = []

    control_resampled = control_samples.copy()
    
    # Define the blocks for bootstrapping
    block_clusters_control = sorted(adata_control.obs[block_key].unique().tolist())

    for block in block_clusters_control:

        block_indices = np.where(adata_control.obs[block_key] == block)[0] # Sample only those cells within the block
        
        block_subsamples = np.random.choice(block_indices, len(block_indices))
        control_resampled[block_indices, :] = control_samples[safe_mask(control_samples, block_subsamples), :]

    for i, pert in enumerate(perturbed_keys):

        pert_resampled = perturbed_samples[i].copy()

        adata_pert = adata_perturbed[adata_perturbed.obs[condition_key] == pert]

        block_clusters_pert = sorted(adata_pert.obs[block_key].unique().tolist())

        for block in block_clusters_pert:

            block_indices = np.where(adata_pert.obs[block_key] == block)[0] # Sample only those cells within the block
            
            block_subsamples = np.random.choice(block_indices, len(block_indices))
            pert_resampled[block_indices, :] = pert_resampled[safe_mask(pert_resampled, block_subsamples), :]

        perturbed_resampled.append(pert_resampled)

    # We need to subset the gene expression matrices for genes with non-zero standard deviation in BOTH cases
    control_resampled_std = control_resampled.std(0)
    perturbed_resampled_std = [sample.std(0) for sample in perturbed_resampled]

    nonzero_gene_indices_control = control_resampled_std.nonzero()[0]
    nonzero_gene_indices_perturbed = [resampled_std.nonzero()[0] for resampled_std in perturbed_resampled_std]

    nonzero_gene_indices = reduce(np.intersect1d, (nonzero_gene_indices_control, *nonzero_gene_indices_perturbed))

    # Subset based on the genes with zero std in both cases
    considered_genes = list([genes[ind] for ind in nonzero_gene_indices])
    
    nodes = set(considered_genes)

    control_resampled = control_resampled[:, nonzero_gene_indices]

    for i, resampled in enumerate(perturbed_resampled):
        perturbed_resampled[i] = resampled[:, nonzero_gene_indices]

    ### Run UT-IGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(control_resampled, invert=True)
    invariance_suffstat = gauss_invariance_suffstat(control_resampled, perturbed_resampled)

    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha_ci)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    ## Run UT-IGSP by considering all possible initial permutations
    setting_list = [dict(known_interventions=[]) for _ in perturbed_resampled]

    if init_perms:

        # Construct random permutations
        inflow_signals = [gene for gene in adata[:, adata.var['type'] == 'bound_complex'].var_names]
        factors = [gene for gene in adata[:, adata.var['type'] == 'factor'].var_names]
        outflow_signals = [gene for gene in adata[:, adata.var['type'] == 'ligand'].var_names]
        
        nodes = set(considered_genes)

        # Convert the permutations in terms of node indices
        converted_perms = []

        for iter in range(n_init_perms):

            converted_perm = []

            # Add inflowing signals
            rm.shuffle(inflow_signals)
            for node in inflow_signals:
                converted_perm.append(considered_genes.index(node))

            # Add GEMs
            rm.shuffle(factors)
            for node in factors:
                converted_perm.append(considered_genes.index(node))

            # Add outflowing signals
            rm.shuffle(outflow_signals)
            for node in outflow_signals:
                converted_perm.append(considered_genes.index(node))

            converted_perms.append(converted_perm)

        est_dag, est_targets_list = unknown_target_igsp(setting_list,
                                                        nodes,
                                                        ci_tester,
                                                        invariance_tester,
                                                        nruns=len(converted_perms),
                                                        initial_permutations=converted_perms)

        # Convert to interventional CPDAG, i.e. I-MEC
        est_icpdag = est_dag.interventional_cpdag(est_targets_list, cpdag=est_dag.cpdag())

        adjacency_cpdag = est_icpdag.to_amat()[0]

        perturbed_targets_list = []

        for i in range(len(est_targets_list)):
            targets_list = list(est_targets_list[i])
            targets_ligand_indices = [list(genes).index(considered_genes[target]) for target in targets_list]
            perturbed_targets_list.append(targets_ligand_indices)
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag,
                'perturbed_targets_indices':perturbed_targets_list}

    else:   

        est_dag, est_targets_list = unknown_target_igsp(setting_list,
                                                        nodes,
                                                        ci_tester,
                                                        invariance_tester,
                                                        nruns=20)

        # Convert to interventional CPDAG, i.e. I-MEC
        est_icpdag = est_dag.interventional_cpdag(est_targets_list, cpdag=est_dag.cpdag())

        adjacency_cpdag = est_icpdag.to_amat()[0]

        perturbed_targets_list = []

        for i in range(len(est_targets_list)):
            targets_list = list(est_targets_list[i])
            targets_ligand_indices = [list(genes).index(considered_genes[target]) for target in targets_list]
            perturbed_targets_list.append(targets_ligand_indices)
            
        return {'nonzero_gene_indices':nonzero_gene_indices,
                'adjacency_cpdag':adjacency_cpdag,
                'perturbed_targets_indices':perturbed_targets_list}

data_directory = '../output/SHH_NEURAL_TUBE/'

# Set number of bootstraps
n_bootstraps = 500
n_jobs = 4
alpha_ci = 0.001
alpha_inv = 0.001

true_pos_edges = [(1, 2),
                (1, 3),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 2),
                (3, 4),
                (3, 5),
                (4, 3),
                (5, 2),
                (5, 3)]

true_neg_edges = [(0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 0),
                (1, 1),
                (1, 4),
                (1, 5),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 0),
                (3, 1),
                (3, 3),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 4),
                (4, 5),
                (5, 0),
                (5, 1),
                (5, 4),
                (5, 5)]

ineligible_rows = [0, 0, 0, 0, 0, 2, 3, 4, 5]
ineligible_cols = [1, 2, 3, 4, 5, 1, 1, 1, 1]

# Load the full dataset to get the celltypes
adata_full = sc.read(data_directory + 'adata_shh_neural_grn.h5ad')
adata = adata_full[:, ['Shh', 'Shh_bound', 'Nkx2-2', 'Olig2', 'Pax6', 'Irx3']]
adata.X = adata.layers['sols'].copy()

genes = list(adata.var_names)

# Use the cell type regions as blocks
condition_key = 'Condition'
control_key = 'WT'
perturbed_keys = ['PERT']
block_key = 'leiden_block'

##################################################################
######        Test non-spatial without perturbation         #####
#################################################################
bagged_adjacency_cpdag = np.zeros((len(genes), len(genes)))

adata_control = adata[adata.obs['Condition'] == 'WT'].copy()

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata_control,
                    False,
                    alpha_ci,
                    20,
                    boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_gsp)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']
    
    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_cpdag[np.ix_(nz_indices, nz_indices)] += adjacency

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_cpdag /= float(n_bootstraps)

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_cpdag.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_cpdag,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_nopert'] = learned_network_results

adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_nopert.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_nopert.csv")

#############################################################################################
######         Test non-spatial without perturbation and initial permutations         ######
############################################################################################
bagged_adjacency_cpdag_perms = np.zeros((len(genes), len(genes)))

adata_control = adata[adata.obs['Condition'] == 'WT'].copy()

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata_control,
                    True,
                    alpha_ci,
                    20,
                    boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_gsp)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_cpdag_perms[np.ix_(nz_indices, nz_indices)] += adjacency

        # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_cpdag[np.ix_(nz_indices, nz_indices)] += adjacency

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_cpdag_perms /= float(n_bootstraps)

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_cpdag_perms.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_cpdag_perms,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_nopert_perms'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_nopert_perms.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_nopert_perms.csv")

##################################################################
######        Test non-spatial using perturbation data        ####
#################################################################
bagged_adjacency_icpdag = np.zeros((len(genes), len(genes)))
bagged_intervention_targets = [np.zeros(len(genes)) for key in perturbed_keys]

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata,
                condition_key,
                control_key,
                perturbed_keys,
                False,
                20,
                alpha_ci,
                alpha_inv,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_utigsp)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']
    int_indices = res['perturbed_targets_indices']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_icpdag[np.ix_(nz_indices, nz_indices)] += adjacency

    # Update the intervention targets
    for i in range(len(int_indices)):

        nonzero_int_indices = int_indices[i]
        intervention_targets = bagged_intervention_targets[i]
        intervention_targets[nonzero_int_indices] += 1
        bagged_intervention_targets[i] = intervention_targets

        # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_icpdag /= float(n_bootstraps)

# Average the intervened targets
for i in range(len(int_indices)):

    intervention_targets = bagged_intervention_targets[i]
    intervention_targets /= float(n_bootstraps) # Average the results
    bagged_intervention_targets[i] = intervention_targets

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_icpdag.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_icpdag,
        'perturbed_targets':bagged_intervention_targets,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_pert'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_pert.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_pert.csv")

####################################################################################
####     Test non-spatial using perturbation data and initial permutations     ####
###################################################################################
bagged_adjacency_icpdag_perms = np.zeros((len(genes), len(genes)))
bagged_intervention_targets_perms = [np.zeros(len(genes)) for key in perturbed_keys]

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata,
                condition_key,
                control_key,
                perturbed_keys,
                True,
                20,
                alpha_ci,
                alpha_inv,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_utigsp)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']
    int_indices = res['perturbed_targets_indices']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_icpdag_perms[np.ix_(nz_indices, nz_indices)] += adjacency

    # Update the intervention targets
    for i in range(len(int_indices)):

        nonzero_int_indices = int_indices[i]
        intervention_targets = bagged_intervention_targets_perms[i]
        intervention_targets[nonzero_int_indices] += 1
        bagged_intervention_targets_perms[i] = intervention_targets

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_icpdag_perms /= float(n_bootstraps)

# Average the intervened targets
for i in range(len(int_indices)):

    intervention_targets = bagged_intervention_targets_perms[i]
    intervention_targets /= float(n_bootstraps) # Average the results
    bagged_intervention_targets_perms[i] = intervention_targets

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_icpdag_perms.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_icpdag_perms,
        'perturbed_targets':bagged_intervention_targets_perms,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_pert_perms'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_pert_perms.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_pert_perms.csv")

##################################################################
######        Test spatial without perturbation         #####
#################################################################
bagged_adjacency_cpdag_spatial = np.zeros((len(genes), len(genes)))

adata_control = adata[adata.obs['Condition'] == 'WT'].copy()

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata_control,
                block_key,
                False,
                alpha_ci,
                20,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_gsp_spatial)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_cpdag_spatial[np.ix_(nz_indices, nz_indices)] += adjacency

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)
# Average the adjacencies
bagged_adjacency_cpdag_spatial /= float(n_bootstraps)

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_cpdag_spatial.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_cpdag_spatial,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_spatial_nopert'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_spatial_nopert.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_spatial_nopert.csv")

######################################################################################
######        Test spatial without perturbation and initial permutations        #####
#####################################################################################
bagged_adjacency_cpdag_spatial_perms= np.zeros((len(genes), len(genes)))

adata_control = adata[adata.obs['Condition'] == 'WT'].copy()

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata_control,
                block_key,
                False,
                alpha_ci,
                20,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_gsp_spatial)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives
accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_cpdag_spatial_perms[np.ix_(nz_indices, nz_indices)] += adjacency

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_cpdag_spatial_perms /= float(n_bootstraps)

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_cpdag_spatial_perms.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_cpdag_spatial_perms,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_spatial_nopert_perms'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_spatial_nopert_perms.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_spatial_nopert_perms.csv")

##################################################################
######        Test spatial using perturbation data         ######
#################################################################
bagged_adjacency_icpdag_spatial = np.zeros((len(genes), len(genes)))
bagged_intervention_targets_spatial = [np.zeros(len(genes)) for key in perturbed_keys]

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata,
                block_key,
                condition_key,
                control_key,
                perturbed_keys,
                False,
                20,
                alpha_ci,
                alpha_inv,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_utigsp_spatial)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives

accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']
    int_indices = res['perturbed_targets_indices']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_icpdag_spatial[np.ix_(nz_indices, nz_indices)] += adjacency

    # Update the intervention targets
    for i in range(len(int_indices)):

        nonzero_int_indices = int_indices[i]
        intervention_targets = bagged_intervention_targets_spatial[i]
        intervention_targets[nonzero_int_indices] += 1
        bagged_intervention_targets_spatial[i] = intervention_targets

    # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_icpdag_spatial /= float(n_bootstraps)

# Average the intervened targets
for i in range(len(int_indices)):

    intervention_targets = bagged_intervention_targets_spatial[i]
    intervention_targets /= float(n_bootstraps) # Average the results
    bagged_intervention_targets_spatial[i] = intervention_targets

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_icpdag_spatial.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_icpdag_spatial,
        'perturbed_targets':bagged_intervention_targets_spatial,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_spatial_pert'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_spatial_pert.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_spatial_pert.csv")

#########################################################################################
######        Test spatial using perturbation data and initial permutations        ######
#########################################################################################
bagged_adjacency_icpdag_spatial_perms = np.zeros((len(genes), len(genes)))
bagged_intervention_targets_spatial_perms = [np.zeros(len(genes)) for key in perturbed_keys]

start = timer()

print(f'starting computations on {n_jobs} cores')

args_allperms = [(adata,
                block_key,
                condition_key,
                control_key,
                perturbed_keys,
                True,
                20,
                alpha_ci,
                alpha_inv,
                boot) for boot in range(n_bootstraps)]
                    
bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_utigsp_spatial)(*arg) for arg in args_allperms)

end = timer()

print(f'elapsed time: {end - start}')

accuracies = np.zeros((n_bootstraps, 4)) # True Positives, False Positives, True Negatives, False Positives

accuracies_filtered = np.zeros((n_bootstraps, 4))

# Sum the results for UT-IGSP with initial permutations
for boot, res in enumerate(bootstrap_results):

    nz_indices = res['nonzero_gene_indices']
    adjacency = res['adjacency_cpdag']
    int_indices = res['perturbed_targets_indices']

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency.nonzero()
    z_rows, z_cols = np.where(adjacency == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies[boot, 0] += 1 # True positive

        else:

            accuracies[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies[boot, 2] += 1 # True negative

        else:

            accuracies[boot, 3] += 1 # False negative (should have been pos)

    # Update the bagged adjacency
    bagged_adjacency_icpdag_spatial_perms[np.ix_(nz_indices, nz_indices)] += adjacency

    # Update the intervention targets
    for i in range(len(int_indices)):

        nonzero_int_indices = int_indices[i]
        intervention_targets = bagged_intervention_targets_spatial_perms[i]
        intervention_targets[nonzero_int_indices] += 1
        bagged_intervention_targets_spatial_perms[i] = intervention_targets

        # Track the filtered accuracy
    adjacency_filtered = adjacency.copy()

    for j in ineligible_rows:
        adjacency_filtered[ineligible_rows[j], ineligible_cols[j]] = 0

    # Benchmark the entries against the skeleton graphs
    nz_rows, nz_cols = adjacency_filtered.nonzero()
    z_rows, z_cols = np.where(adjacency_filtered == 0)

    for i, row in enumerate(nz_rows):

        col = nz_cols[i]

        if (row, col) in true_pos_edges:

            accuracies_filtered[boot, 0] += 1 # True positive

        else:

            accuracies_filtered[boot, 1] += 1 # False positive

    for i, row in enumerate(z_rows):

        col = z_cols[i]

        if (row, col) in true_neg_edges:

            accuracies_filtered[boot, 2] += 1 # True negative

        else:

            accuracies_filtered[boot, 3] += 1 # False negative (should have been pos)

# Average the adjacencies
bagged_adjacency_icpdag_spatial_perms /= float(n_bootstraps)

# Average the intervened targets
for i in range(len(int_indices)):

    intervention_targets = bagged_intervention_targets_spatial_perms[i]
    intervention_targets /= float(n_bootstraps) # Average the results
    bagged_intervention_targets_spatial_perms[i] = intervention_targets

learned_edges = []

nonzero_rows, nonzero_cols = bagged_adjacency_icpdag_spatial_perms.nonzero()

for i in range(len(nonzero_rows)):
    
    learned_edges.append((genes[nonzero_rows[i]], genes[nonzero_cols[i]]))

learned_network_results =  {'genes': genes,
        'causal_adjacency': bagged_adjacency_icpdag_spatial_perms,
        'perturbed_targets':bagged_intervention_targets_spatial_perms,
        'networks':{'joined':{'nodes':genes, 'edges':learned_edges}}}

# Store the results
adata_full.uns['causal_network_spatial_pert_perms'] = learned_network_results
adata_full.write(data_directory + 'adata_shh_neural_grn.h5ad', compression='gzip')

accuracies_df = pd.DataFrame(accuracies)
accuracies_df.to_csv(data_directory + "accuracies_shh_neural_grn_spatial_pert_perms.csv")

accuracies_filtered_df = pd.DataFrame(accuracies_filtered)
accuracies_filtered_df.to_csv(data_directory + "accuracies_filtered_shh_neural_grn_spatial_pert_perms.csv")

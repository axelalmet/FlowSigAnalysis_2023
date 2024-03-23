from cnmf import cNMF
import scanpy as sc
import numpy as np

adata_kang = sc.read_h5ad('../data/kang18_counts_25k.h5ad')

# Prepare for cNMF
adata_kang_cnmf = adata_kang.copy()
adata_kang_cnmf.X = adata_kang_cnmf.layers['counts']

numiter=200 # Number of NMF replicates. Set this to a larger value ~200 for real data. We set this to a relatively low value here for illustration at a faster speed
numhvgenes=2000 ## Number of over-dispersed genes to use for running the actual gemizations

## Results will be saved to [output_directory]/[run_name] which in this example is example_PBMC/cNMF/pbmc_cNMF
output_directory = data_directory + 'Kang2018/cNMF'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
run_name = 'kang18_counts_25k_cNMF'

## Specify the Ks to use as a space separated list in this case "5 6 7 8 9 10"
K = ' '.join([str(i) for i in range(6, 20, 2)])

## To speed this up, you can run it for only K=7-8 with the option below
#K = ' '.join([str(i) for i in range(7,9)])

seed = 0 ## Specify a seed pseudorandom number generation for reproducibility

## Path to the filtered counts dataset we output previously
countfn = '../data/kang18_counts_25k_cnmf.h5ad'

## Initialize the cnmf object that will be used to run analyses
cnmf_obj = cNMF(output_dir=output_directory, name=run_name)

## Prepare the data, I.e. subset to 2000 high-variance genes, and variance normalize
cnmf_obj.prepare(counts_fn=countfn, components=np.arange(6, 20, 2), n_iter=numiter, seed=seed, num_highvar_genes=numhvgenes)

## Specify that the jobs are being distributed over a single worker (total_workers=1) and then launch that worker
cnmf_obj.gemize(worker_i=0, total_workers=4)

usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.load_results(K=8, density_threshold=0.2)
usage_norm.columns = ['cNMF_%d' % i for i in usage_norm.columns]

adata_kang.uns['cnmf_info'] = {'usage_norm': usage_norm, 
                            'gep_scores': gep_scores,
                            'gep_tpm': gep_tpm,
                            'topgenes': topgenes,
                            'n_gems': 8,
                            'vars': gep_scores.index.tolist()}
                            

adata_kang.write('../data/kang18_counts_25k.h5ad', compression='gzip')
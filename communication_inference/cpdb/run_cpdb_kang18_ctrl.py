import pandas as pd
from cellphonedb.src.core.methods import cpdb_degs_analysis_method

data_directory = '../../data/'
cpdb_file_path = data_directory + 'cellphonedb.zip'
meta_file_path = data_directory + 'kang18_counts_25k_metadata_ctrl.tsv'
counts_file_path = data_directory + 'kang18_counts_25k_ctrl.h5ad'
degs_file_path = data_directory + 'kang18_counts_25k_degs_ctrl.tsv'
active_tf_path = data_directory + 'kang18_counts_25k_active_tfs_ctrl.tsv'
out_path = data_directory + 'kang18_counts_25k_ctrl_cpdb_method3'

cpdb_results = cpdb_degs_analysis_method.call(
    cpdb_file_path = cpdb_file_path,                            # mandatory: CellphoneDB database zip file.
    meta_file_path = meta_file_path,                            # mandatory: tsv file defining barcodes to cell label.
    counts_file_path = counts_file_path,                        # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
    degs_file_path = degs_file_path,                            # mandatory: tsv file with DEG to account.
    counts_data = 'hgnc_symbol',                                # defines the gene annotation in counts matrix.
    active_tfs_file_path = active_tf_path,                      # optional: defines cell types and their active TFs.
    score_interactions = True,                                  # optional: whether to score interactions or not. 
    threshold = 0.1,                                            # defines the min % of cells expressing a gene for this to be employed in the analysis.
    result_precision = 3,                                       # Sets the rounding for the mean values in significan_means.
    separator = '|',                                            # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
    debug = False,                                              # Saves all intermediate tables emplyed during the analysis in pkl format.
    output_path = out_path,                                     # Path to save results
    output_suffix = None,                                       # Replaces the timestamp in the output files by a user defined string in the  (default: None)
    threads = 4
    )
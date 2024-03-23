library(scITD)
library(zellkonverter)
library(SingleCellExperiment)
library(dplyr)
library(stringr)

kang.sce <- readH5AD('../data/kang18_counts_25k.h5ad')

kang.sce[, kang.sce$cell_abbr != "Mega"] # Filter out Megakaryocytes

kang.counts <- assay(kang.sce, "counts")
kang.genes <- rownames(kang.sce)
kang.meta <- data.frame(colData(kang.sce))

# Generate donors and labels columns
kang.meta$ctypes <- as.factor(droplevels(kang.meta$cell_abbr))
kang.meta$donors <- as.factor(str_replace(kang.meta$sample, '&', ''))
kang.meta$stim <- as.factor(kang.meta$condition)

param_list <- initialize_params(ctypes_use = c('CD4T', 'CD14', 'B', 'NK', 'CD8T', 'FGR3', 'DCs'),
                                ncores = 1)

kang_container <- make_new_container(count_data=kang.counts, 
                                     meta_data=kang.meta,
                                     gn_convert = data.frame(kang.genes, kang.genes), 
                                     params=param_list,
                                     label_donor_sex = FALSE)

kang_container <- form_tensor(kang_container, donor_min_cells=3,
                              norm_method='trim', scale_factor=10000,
                              vargenes_method='norm_var', vargenes_thresh=500,
                              scale_var = TRUE, var_scale_power = 0.5)

kang_container <- run_tucker_ica(kang_container, ranks=c(2, 3),
                                 tucker_type = 'regular', rotation_type = 'hybrid')

kang_container <- get_meta_associations(kang_container, vars_test=c('condition'), stat_use='pval')

# plot donor scores
kang_container <- plot_donor_matrix(kang_container, meta_vars=c('condition'),
                                    show_donor_ids = TRUE,
                                    add_meta_associations='pval')

kang_container$plots$donor_matrix

# get significant genes
kang_container <- get_lm_pvals(kang_container)

# generate the loadings plots
kang_container <- get_all_lds_factor_plots(kang_container, 
                                           use_sig_only=TRUE,
                                           nonsig_to_zero=TRUE,
                                           sig_thresh=.05,
                                           display_genes=FALSE,
                                           gene_callouts = TRUE,
                                           callout_n_gene_per_ctype = 5,
                                           show_var_explained = TRUE)

# arrange the plots into a figure and show the figure
myfig <- render_multi_plots(kang_container,data_type='loadings')
myfig

f1_data <- get_one_factor(kang_container, factor_select=1)
f1_dscores <- f1_data[[1]]
f1_loadings <- f1_data[[2]]

print(head(f1_dscores))

f1_pvals <- get_one_factor_gene_pvals(kang_container, factor_select=1)

print(head(f1_pvals))

kang_container <- run_gsea_one_factor(kang_container, factor_select=1, method="fgsea", thresh=0.05, db_use=c("GO"), signed=TRUE)

print(head(f1_pvals))


kang_container <- determine_ranks_tucker(kang_container, max_ranks_test=c(10,15),
                                         shuffle_level='cells', 
                                         num_iter=10, 
                                         norm_method='trim',
                                         scale_factor=10000,
                                         scale_var=TRUE,
                                         var_scale_power=0.5)

kang_container$plots$rank_determination_plot


lr_network = readRDS(url("https://zenodo.org/record/3260758/files/lr_network.rds"))
lr_network = lr_network %>% mutate(bonafide = ! database %in% c("ppi_prediction","ppi_prediction_go"))
lr_network = lr_network %>% dplyr::rename(ligand = from, receptor = to) %>% distinct(ligand, receptor, bonafide)

# structuring the data properly
lr_pairs <- as.matrix(lr_network)
lr_pairs <- lr_pairs[lr_pairs[,'bonafide']=='TRUE',]
lr_pairs <- lr_pairs[,c(1,2)]

kang_container <- prep_LR_interact(kang_container, lr_pairs, norm_method='trim', scale_factor=10000,
                                   var_scale_power=0.5)

sft_thresh <- c(20, 20, 16, 12, 20, 14, 16)

invisible({capture.output({
  kang_container <- get_gene_modules(kang_container,sft_thresh)
})})

invisible({capture.output({
  lr_hmap <- compute_LR_interact(kang_container, lr_pairs, sig_thresh=.0001,
                                 percentile_exp_rec=0.95, add_ld_fact_sig=TRUE)
})})

lr_hmap

pdf(file="kang18_scitd_lr_heatmap.pdf",
    width=18, height=40)
ComplexHeatmap::draw(lr_hmap)
dev.off()


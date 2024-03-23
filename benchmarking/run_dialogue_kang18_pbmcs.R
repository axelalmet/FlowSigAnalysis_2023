library(DIALOGUE)
library(zellkonverter)
library(SingleCellExperiment)


DLG.get.file<-function(file1){
  workdir<-"./"
  return(paste0(workdir,file1))
}

kang.sce <- readH5AD(paste(data_directory, '../data/kang18_counts_25k.h5ad', sep=''))

kang.sce <- kang.sce[, kang.sce$cell_abbr != "Mega"] # Filter out Megakaryocytes

kang.exp <- assay(kang.sce, "X")
kang.exp <- as.matrix(kang.exp)

kang.pca <- reducedDim(kang.sce, "X_pca")
colnames(kang.pca) <- paste0("PC", 1:dim(kang.pca)[2])

kang.meta <- as.data.frame(colData(kang.sce))
kang.meta$pheno <- FALSE
kang.meta$pheno[kang.meta$condition == "stim"] <- TRUE
kang.meta$pheno <- as.factor(kang.meta$pheno)

kang.celltypes <- c('CD14', 'CD4T', 'DCs', 'NK', 'CD8T', 'B', 'FGR3')

kang.samples <- kang.meta$sample

kang.cellQ = kang.sce$nFeature_RNA
names(kang.cellQ) <- row.names(kang.meta)

dialogue.kang.cell.type <- list()

# Create the dialogue cell type object for each major cell type
dialogue.kang.cell.type$CD14  <- make.cell.type(name="CD14",
                                                tpm=kang.exp[, kang.meta$cell_abbr == "CD14"],
                                                samples=kang.meta$sample[kang.meta$cell_abbr == "CD14"],
                                                X=kang.pca[kang.meta$cell_abbr == "CD14", ],
                                                metadata=kang.meta[kang.meta$cell_abbr == "CD14", ],
                                                cellQ=kang.cellQ[kang.meta$cell_abbr == "CD14"])

dialogue.kang.cell.type$CD4T  <- make.cell.type(name="CD4T",
                                                tpm=kang.exp[, kang.meta$cell_abbr == "CD4T"],
                                                samples=kang.meta$sample[kang.meta$cell_abbr == "CD4T"],
                                                X=kang.pca[kang.meta$cell_abbr == "CD4T", ],
                                                metadata=kang.meta[kang.meta$cell_abbr == "CD4T", ],
                                                cellQ=kang.cellQ[kang.meta$cell_abbr == "CD4T"])

dialogue.kang.cell.type$DCs  <- make.cell.type(name="DCs",
                                               tpm=kang.exp[, kang.meta$cell_abbr == "DCs"],
                                               samples=kang.meta$sample[kang.meta$cell_abbr == "DCs"],
                                               X=kang.pca[kang.meta$cell_abbr == "DCs", ],
                                               metadata=kang.meta[kang.meta$cell_abbr == "DCs", ],
                                               cellQ=kang.cellQ[kang.meta$cell_abbr == "DCs"])

dialogue.kang.cell.type$NK  <- make.cell.type(name="NK",
                                              tpm=kang.exp[, kang.meta$cell_abbr == "NK"],
                                              samples=kang.meta$sample[kang.meta$cell_abbr == "NK"],
                                              X=kang.pca[kang.meta$cell_abbr == "NK", ],
                                              metadata=kang.meta[kang.meta$cell_abbr == "NK", ],
                                              cellQ=kang.cellQ[kang.meta$cell_abbr == "NK"])

dialogue.kang.cell.type$CD8T  <- make.cell.type(name="CD8T",
                                                tpm=kang.exp[, kang.meta$cell_abbr == "CD8T"],
                                                samples=kang.meta$sample[kang.meta$cell_abbr == "CD8T"],
                                                X=kang.pca[kang.meta$cell_abbr == "CD8T", ],
                                                metadata=kang.meta[kang.meta$cell_abbr == "CD8T", ],
                                                cellQ=kang.cellQ[kang.meta$cell_abbr == "CD8T"])

dialogue.kang.cell.type$B  <- make.cell.type(name="B",
                                             tpm=kang.exp[, kang.meta$cell_abbr == "B"],
                                             samples=kang.meta$sample[kang.meta$cell_abbr == "B"],
                                             X=kang.pca[kang.meta$cell_abbr == "B", ],
                                             metadata=kang.meta[kang.meta$cell_abbr == "B", ],
                                             cellQ=kang.cellQ[kang.meta$cell_abbr == "B"])

dialogue.kang.cell.type$FGR3  <- make.cell.type(name="FGR3",
                                                tpm=kang.exp[, kang.meta$cell_abbr == "FGR3"],
                                                samples=kang.meta$sample[kang.meta$cell_abbr == "FGR3"],
                                                X=kang.pca[kang.meta$cell_abbr == "FGR3", ],
                                                metadata=kang.meta[kang.meta$cell_abbr == "FGR3", ],
                                                cellQ=kang.cellQ[kang.meta$cell_abbr == "FGR3"])

dialogue.kang.param <- DLG.get.param(k = 4,
                                     conf = c("cellQ"),
                                     covar=c("cellQ", "tme.qc"),
                                     results.dir = DLG.get.file("DLG_Results/"),
                                     pheno="condition")

R<-DIALOGUE.run(rA = dialogue.kang.cell.type, # list of cell.type objects
                main = "kang18_run_K_4",
                param = dialogue.kang.param,
                plot.flag = F)

saveRDS(R, file="DLG_Results/DLG.full.output_kang18_run_K_4.rds")

R <- readRDS(file="DLG_Results/DLG.full.output_kang18_run_K_4.rds")

kang.mark.samples <- c("stim&1016", "stim&1039", "stim&107", "stim&1244", "stim&1256", "stim&1488")

DIALOGUE.plot(R,results.dir = "DLG_Results/",filename="DLG.full.output_kang18_run_K_4_MCP1",
              pheno = "pathology" ,mark.samples = kang.mark.samples,  MCPs=1)

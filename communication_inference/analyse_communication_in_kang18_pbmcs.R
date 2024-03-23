# Load the relevant packages
library(dplyr)
library(CellChat)
library(SummarizedExperiment)
library(zellkonverter)

### Set the ligand-receptor database. Here we will use the "Secreted signalling" database for cell-cell communication (let's look at ECM-receptor in the future )
CellChatDB <- CellChatDB.human # The othe roption is CellChatDB.human
showDatabaseCategory(CellChatDB)
CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling") # Other options include ECM-Receptor and Cell-Cell Contact

### Kang et al. (2018) ###
kang_sce <- readH5AD("../data/kang18_counts_25k.h5ad")

kang.data.input <- assay(kang_sce, "X")
kang.meta.data <- data.frame(colData(kang_sce))

# kang.data.input <- kang.data.input - min(kang.data.input) # Need to make sure the data isn't negative (it is in this case)
# access meta data
kang.identity <- data.frame(group = kang.meta.data$cell_abbr, row.names = row.names(kang.meta.data))

ctrl.use <- rownames(kang.meta.data)[kang.meta.data$condition == "ctrl"] # extract the cell names from disease data
stim.use <- rownames(kang.meta.data)[kang.meta.data$condition == "stim"] # extract the cell names from disease data

ctrl.data.input <- kang.data.input[, ctrl.use]
stim.data.input <- kang.data.input[, stim.use]

ctrl.meta <- kang.meta.data[kang.meta.data$condition == "ctrl", ]
stim.meta <- kang.meta.data[kang.meta.data$condition == "stim", ]

kang.ctrl.identity <- data.frame(group = ctrl.meta$cell_abbr, row.names = row.names(ctrl.meta))
kang.stim.identity <- data.frame(group = stim.meta$cell_abbr, row.names = row.names(stim.meta))

# Create the cellchat objects
kang.ctrl.cc <-createCellChat(object = ctrl.data.input, do.sparse = T, meta = kang.ctrl.identity, group.by = "group")
levels(kang.ctrl.cc@idents) # show factor levels of the cell labels

kang.stim.cc <-createCellChat(object = stim.data.input, do.sparse = T, meta = kang.stim.identity, group.by = "group")
levels(kang.stim.cc@idents) # show factor levels of the cell labels

ctrlGroupSize <- as.numeric(table(kang.ctrl.cc@idents)) # Get the number of cells in each group
stimGroupSize <- as.numeric(table(kang.stim.cc@idents)) # Get the number of cells in each group

min_pct <- 0.1
ctrlMinGroupSize <- round(min_pct / 100 * sum(ctrlGroupSize))
stimMinGroupSize <- round(min_pct / 100 * sum(stimGroupSize))

# Set the databases
kang.ctrl.cc@DB <- CellChatDB.use 
kang.stim.cc@DB <- CellChatDB.use 

# We now identify over-expressed ligands/receptors in a cell group and then project gene expression data onto the protein-protein interaction network
### ctrl
kang.ctrl.cc <- subsetData(kang.ctrl.cc) # We subset the expression data of signalling genes to save on computational cost
kang.ctrl.cc <- identifyOverExpressedGenes(kang.ctrl.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
kang.ctrl.cc <- identifyOverExpressedInteractions(kang.ctrl.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
kang.ctrl.cc <- projectData(kang.ctrl.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

kang.ctrl.cc <- computeCommunProb(kang.ctrl.cc, raw.use = FALSE, population.size = FALSE, type='triMean')
kang.ctrl.cc <- filterCommunication(kang.ctrl.cc, min.cells = ctrlMinGroupSize)

kang.ctrl.cc <- computeCommunProbPathway(kang.ctrl.cc) # Calculate the probabilities at the signalling level
kang.ctrl.cc <- aggregateNet(kang.ctrl.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

### stim
kang.stim.cc <- subsetData(kang.stim.cc) # We subset the expression data of signalling genes to save on computational cost
kang.stim.cc <- identifyOverExpressedGenes(kang.stim.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
kang.stim.cc <- identifyOverExpressedInteractions(kang.stim.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
kang.stim.cc <- projectData(kang.stim.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

kang.stim.cc <- computeCommunProb(kang.stim.cc, raw.use = FALSE, population.size = FALSE)
kang.stim.cc <- filterCommunication(kang.stim.cc, min.cells = stimMinGroupSize)

kang.stim.cc <- computeCommunProbPathway(kang.stim.cc) # Calculate the probabilities at the signalling level
kang.stim.cc <- aggregateNet(kang.stim.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

kang.ctrl.net <- subsetCommunication(kang.ctrl.cc)
kang.stim.net <- subsetCommunication(kang.stim.cc)

write.csv(kang.ctrl.net, file = "./output/kang18_counts_25k_communications_ctrl.csv")
write.csv(kang.stim.net, file = "./output/kang18_counts_25k_communications_stim.csv")


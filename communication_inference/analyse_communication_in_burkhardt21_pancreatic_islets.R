# Load the relevant packages
library(dplyr)
library(CellChat)
library(SummarizedExperiment)
library(zellkonverter)

### Set the ligand-receptor database. Here we will use the "Secreted signalling" database for cell-cell communication (let's look at ECM-receptor in the future )
CellChatDB <- CellChatDB.human # The othe roption is CellChatDB.human
showDatabaseCategory(CellChatDB)
CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling") # Other options include ECM-Receptor and Cell-Cell Contact

### Burkhardt et al. (2021)
burkhardt_sce <- readH5AD("../data/burkhardt21_merged.h5ad")

burkhardt.data.input <- assay(burkhardt_sce, "X")

burkhardt.meta.data <- data.frame(colData(burkhardt_sce))

#### First by merged cell types ####
burkhardt.identity <- data.frame(group = burkhardt.meta.data$Type, row.names = row.names(burkhardt.meta.data))

control.use <- rownames(burkhardt.meta.data)[burkhardt.meta.data$Condition == "Ctrl"] # extract the cell names from disease data
stim.use <- rownames(burkhardt.meta.data)[burkhardt.meta.data$Condition == "IFNg"] # extract the cell names from disease data

control.data.input <- burkhardt.data.input[, control.use]
stim.data.input <- burkhardt.data.input[, stim.use]

control.meta <- burkhardt.meta.data[burkhardt.meta.data$Condition == "Ctrl", ]
stim.meta <- burkhardt.meta.data[burkhardt.meta.data$Condition == "IFNg", ]

burkhardt.control.identity <- data.frame(group = control.meta$Type, row.names = row.names(control.meta))
burkhardt.stim.identity <- data.frame(group = stim.meta$Type, row.names = row.names(stim.meta))

# Create the cellchat objects
burkhardt.control.cc <-createCellChat(object = control.data.input, do.sparse = T, meta = burkhardt.control.identity, group.by = "group")
levels(burkhardt.control.cc@idents) # show factor levels of the cell labels

burkhardt.stim.cc <-createCellChat(object = stim.data.input, do.sparse = T, meta = burkhardt.stim.identity, group.by = "group")
levels(burkhardt.stim.cc@idents) # show factor levels of the cell labels

controlGroupSize <- as.numeric(table(burkhardt.control.cc@idents)) # Get the number of cells in each group
stimGroupSize <- as.numeric(table(burkhardt.stim.cc@idents)) # Get the number of cells in each group

# Set the databases
burkhardt.control.cc@DB <- CellChatDB.use 
burkhardt.stim.cc@DB <- CellChatDB.use 

# We now identify over-expressed ligands/receptors in a cell group and then project gene expression data onto the protein-protein interaction network
### Control
burkhardt.control.cc <- subsetData(burkhardt.control.cc) # We subset the expression data of signalling genes to save on computational cost
burkhardt.control.cc <- identifyOverExpressedGenes(burkhardt.control.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
burkhardt.control.cc <- identifyOverExpressedInteractions(burkhardt.control.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
burkhardt.control.cc <- projectData(burkhardt.control.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

burkhardt.control.cc <- computeCommunProb(burkhardt.control.cc, raw.use = FALSE, population.size = FALSE)
burkhardt.control.cc <- computeCommunProbPathway(burkhardt.control.cc) # Calculate the probabilities at the signalling level
burkhardt.control.cc <- aggregateNet(burkhardt.control.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

### Stimulated
burkhardt.stim.cc <- subsetData(burkhardt.stim.cc) # We subset the expression data of signalling genes to save on computational cost
burkhardt.stim.cc <- identifyOverExpressedGenes(burkhardt.stim.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
burkhardt.stim.cc <- identifyOverExpressedInteractions(burkhardt.stim.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
burkhardt.stim.cc <- projectData(burkhardt.stim.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

burkhardt.stim.cc <- computeCommunProb(burkhardt.stim.cc, raw.use = FALSE, population.size = FALSE)
burkhardt.stim.cc <- computeCommunProbPathway(burkhardt.stim.cc) # Calculate the probabilities at the signalling level
burkhardt.stim.cc <- aggregateNet(burkhardt.stim.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

burkhardt.control.net <- subsetCommunication(burkhardt.control.cc)
burkhardt.stim.net <- subsetCommunication(burkhardt.stim.cc)

write.csv(burkhardt.control.net, file = "/Users/axelalmet/Documents/scRNASeqAnalysisAndModelling/CellChat/Output/Burkhardt2021/burkhardt21_merged_communications_Ctrl.csv")
write.csv(burkhardt.stim.net, file = "/Users/axelalmet/Documents/scRNASeqAnalysisAndModelling/CellChat/Output/Burkhardt2021/burkhardt21_merged_communications_IFNg.csv")

#### Now by the original clusters ####
burkhardt.identity <- data.frame(group = burkhardt.meta.data$leiden, row.names = row.names(burkhardt.meta.data))

control.use <- rownames(burkhardt.meta.data)[burkhardt.meta.data$Condition == "Ctrl"] # extract the cell names from disease data
stim.use <- rownames(burkhardt.meta.data)[burkhardt.meta.data$Condition == "IFNg"] # extract the cell names from disease data

control.data.input <- burkhardt.data.input[, control.use]
stim.data.input <- burkhardt.data.input[, stim.use]

control.meta <- burkhardt.meta.data[burkhardt.meta.data$Condition == "Ctrl", ]
stim.meta <- burkhardt.meta.data[burkhardt.meta.data$Condition == "IFNg", ]

burkhardt.control.identity <- data.frame(group = control.meta$leiden, row.names = row.names(control.meta))
burkhardt.stim.identity <- data.frame(group = stim.meta$leiden, row.names = row.names(stim.meta))

# Create the cellchat objects
burkhardt.control.cc <-createCellChat(object = control.data.input, do.sparse = T, meta = burkhardt.control.identity, group.by = "group")
levels(burkhardt.control.cc@idents) # show factor levels of the cell labels

burkhardt.stim.cc <-createCellChat(object = stim.data.input, do.sparse = T, meta = burkhardt.stim.identity, group.by = "group")
levels(burkhardt.stim.cc@idents) # show factor levels of the cell labels

controlGroupSize <- as.numeric(table(burkhardt.control.cc@idents)) # Get the number of cells in each group
stimGroupSize <- as.numeric(table(burkhardt.stim.cc@idents)) # Get the number of cells in each group

# Set the databases
burkhardt.control.cc@DB <- CellChatDB.use 
burkhardt.stim.cc@DB <- CellChatDB.use 

# We now identify over-expressed ligands/receptors in a cell group and then project gene expression data onto the protein-protein interaction network
### Control
burkhardt.control.cc <- subsetData(burkhardt.control.cc) # We subset the expression data of signalling genes to save on computational cost
burkhardt.control.cc <- identifyOverExpressedGenes(burkhardt.control.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
burkhardt.control.cc <- identifyOverExpressedInteractions(burkhardt.control.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
burkhardt.control.cc <- projectData(burkhardt.control.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

burkhardt.control.cc <- computeCommunProb(burkhardt.control.cc, raw.use = FALSE, population.size = FALSE)
burkhardt.control.cc <- computeCommunProbPathway(burkhardt.control.cc) # Calculate the probabilities at the signalling level
burkhardt.control.cc <- aggregateNet(burkhardt.control.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

### Stimulated
burkhardt.stim.cc <- subsetData(burkhardt.stim.cc) # We subset the expression data of signalling genes to save on computational cost
burkhardt.stim.cc <- identifyOverExpressedGenes(burkhardt.stim.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
burkhardt.stim.cc <- identifyOverExpressedInteractions(burkhardt.stim.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
burkhardt.stim.cc <- projectData(burkhardt.stim.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

burkhardt.stim.cc <- computeCommunProb(burkhardt.stim.cc, raw.use = FALSE, population.size = FALSE)
burkhardt.stim.cc <- computeCommunProbPathway(burkhardt.stim.cc) # Calculate the probabilities at the signalling level
burkhardt.stim.cc <- aggregateNet(burkhardt.stim.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

burkhardt.control.net <- subsetCommunication(burkhardt.control.cc)
burkhardt.stim.net <- subsetCommunication(burkhardt.stim.cc)

write.csv(burkhardt.control.net, file = "./output/burkhardt21_leiden_communications_Ctrl.csv")
write.csv(burkhardt.stim.net, file = "./output/burkhardt21_leiden_communications_IFNg.csv")


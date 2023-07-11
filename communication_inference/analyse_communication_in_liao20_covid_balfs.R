# Load the relevant packages
library(dplyr)
library(CellChat)
library(SummarizedExperiment)
library(zellkonverter)

### Set the ligand-receptor database. Here we will use the "Secreted signalling" database for cell-cell communication (let's look at ECM-receptor in the future )
CellChatDB <- CellChatDB.human # The othe roption is CellChatDB.human
showDatabaseCategory(CellChatDB)
CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling") # Other options include ECM-Receptor and Cell-Cell Contact

### Liao et al. (2020) ###
liao_sce <- readH5AD("../data/liao20_sub.h5ad")

liao.data.input <- assay(liao_sce, "X")
liao.meta.data <- data.frame(colData(liao_sce))

# lee.data.input <- lee.data.input - min(lee.data.input) # Need to make sure the data isn't negative (it is in this case)
# access meta data
liao.identity <- data.frame(group = liao.meta.data$celltype, row.names = row.names(liao.meta.data))

healthy.use <- rownames(liao.meta.data)[liao.meta.data$group == "HC"] # extract the cell names from disease data
moderate.use <- rownames(liao.meta.data)[liao.meta.data$group == "M"] # extract the cell names from disease data
severe.use <- rownames(liao.meta.data)[liao.meta.data$group == "S"] # extract the cell names from disease data

healthy.data.input <- liao.data.input[, healthy.use]
moderate.data.input <- liao.data.input[, moderate.use]
severe.data.input <- liao.data.input[, severe.use]

healthy.meta <- liao.meta.data[liao.meta.data$group == "HC", ]
moderate.meta <- liao.meta.data[liao.meta.data$group == "M", ]
severe.meta <- liao.meta.data[liao.meta.data$group == "S", ]

healthy.identity <- data.frame(group = healthy.meta$celltype, row.names = row.names(healthy.meta))
moderate.identity <- data.frame(group = moderate.meta$celltype, row.names = row.names(moderate.meta))
severe.identity <- data.frame(group = severe.meta$celltype, row.names = row.names(severe.meta))

# Create the cellchat objects
healthy.cc <-createCellChat(object = healthy.data.input, do.sparse = T, meta = healthy.identity, group.by = "group")
levels(healthy.cc@idents) # show factor levels of the cell labels

moderate.cc <-createCellChat(object = moderate.data.input, do.sparse = T, meta = moderate.identity, group.by = "group")
levels(moderate.cc@idents) # show factor levels of the cell labels

severe.cc <-createCellChat(object = severe.data.input, do.sparse = T, meta = severe.identity, group.by = "group")
levels(severe.cc@idents) # show factor levels of the cell labels

healthy.cc@idents <- droplevels(healthy.cc@idents, exclude = setdiff(levels(healthy.cc@idents),unique(healthy.cc@idents)))
moderate.cc@idents <- droplevels(moderate.cc@idents, exclude = setdiff(levels(moderate.cc@idents),unique(moderate.cc@idents)))
severe.cc@idents <- droplevels(severe.cc@idents, exclude = setdiff(levels(severe.cc@idents),unique(severe.cc@idents)))

healthyGroupSize <- as.numeric(table(healthy.cc@idents)) # Get the number of cells in each group
moderateGroupSize <- as.numeric(table(moderate.cc@idents)) # Get the number of cells in each group
severeGroupSize <- as.numeric(table(severe.cc@idents)) # Get the number of cells in each group

# Set the databases
healthy.cc@DB <- CellChatDB.use 
moderate.cc@DB <- CellChatDB.use 
severe.cc@DB <- CellChatDB.use 

# We now identify over-expressed ligands/receptors in a cell group and then project gene expression data onto the protein-protein interaction network
### Healthy
healthy.cc <- subsetData(healthy.cc) # We subset the expression data of signalling genes to save on computational cost
healthy.cc <- identifyOverExpressedGenes(healthy.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
healthy.cc <- identifyOverExpressedInteractions(healthy.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
healthy.cc <- projectData(healthy.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

healthy.cc <- computeCommunProb(healthy.cc, raw.use = FALSE, population.size = FALSE)
healthy.cc <- filterCommunication(healthy.cc, min.cells = 0.001*sum(healthyGroupSize))
healthy.cc <- computeCommunProbPathway(healthy.cc) # Calculate the probabilities at the signalling level
healthy.cc <- aggregateNet(healthy.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

### Moderate
moderate.cc <- subsetData(moderate.cc) # We subset the expression data of signalling genes to save on computational cost
moderate.cc <- identifyOverExpressedGenes(moderate.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
moderate.cc <- identifyOverExpressedInteractions(moderate.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
moderate.cc <- projectData(moderate.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

moderate.cc <- computeCommunProb(moderate.cc, raw.use = FALSE, population.size = FALSE)
moderate.cc <- filterCommunication(moderate.cc, min.cells = 0.001*sum(moderateGroupSize))
moderate.cc <- computeCommunProbPathway(moderate.cc) # Calculate the probabilities at the signalling level
moderate.cc <- aggregateNet(moderate.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

### Severe
severe.cc <- subsetData(severe.cc) # We subset the expression data of signalling genes to save on computational cost
severe.cc <- identifyOverExpressedGenes(severe.cc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
severe.cc <- identifyOverExpressedInteractions(severe.cc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
severe.cc <- projectData(severe.cc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.

severe.cc <- computeCommunProb(severe.cc, raw.use = FALSE, population.size = FALSE)
severe.cc <- filterCommunication(severe.cc, min.cells = 0.001*sum(severeGroupSize))
severe.cc <- computeCommunProbPathway(severe.cc) # Calculate the probabilities at the signalling level
severe.cc <- aggregateNet(severe.cc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities

healthy.net <- subsetCommunication(healthy.cc)
moderate.net <- subsetCommunication(moderate.cc)
severe.net <- subsetCommunication(severe.cc)

write.csv(healthy.net, file = "./output/liao20_sub_communications_HC.csv")
write.csv(moderate.net, file = "./output/liao20_sub_communications_M.csv")
write.csv(severe.net, file = "./output/liao20_sub_communications_S.csv")


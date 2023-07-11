# Process brain data
library(Seurat)
library(SummarizedExperiment)
library(zellkonverter)
library(dplyr)
library(CellChat)

### Set the ligand-receptor database. Here we will use the "Secreted signalling" database for cell-cell communication (let's look at ECM-receptor in the future )
CellChatDB <- CellChatDB.human # The othe roption is CellChatDB.mouse
showDatabaseCategory(CellChatDB)
CellChatDB.use <- subsetDB(CellChatDB, search = c("Secreted Signaling")) # Other options include ECM-Receptor and Cell-Cell Contact

min_percentage <- 0.1

data_directory <- "../data/"

samples <- c("D18", "D35")

organoid.merged <- readH5AD(paste(data_directory, "brain_organoid_D18_D35.h5ad", sep=""))

for (sample in samples)
{
  
  organoid.sce <- organoid.merged[organoid.merged$day == sample, ]
  
  levels(organoid.sce$ident) <- c("A", "B", 'C', "D", "E", "F") # For some reason, CellChat doesn't like it when you use '0' for cell type labels
  
  # Split up the data here
  organoid.data.input <- assay(organoid.sce, "X")
  organoid.meta.data <- data.frame(colData(organoid.sce))
  
  ### Analyse fibrotic data first
  organoid.meta.data$ident <- droplevels(organoid.sce$ident)
  
  organoid.identity <- data.frame(group = organoid.sce$ident, row.names = row.names(organoid.meta.data))
  
  # Create the cellchat object
  organoid.ccc <- createCellChat(object = organoid.data.input, do.sparse = T, meta = organoid.identity, group.by = "group")
  levels(organoid.ccc@idents) # show factor levels of the cell labels
  
  organoidGroupSize <- as.numeric(table(organoid.ccc@idents)) # Get the number of cells in each group
  organoidMinCells <- floor(min_percentage / 100.0 * sum(organoidGroupSize))
  
  organoid.ccc@DB <- CellChatDB.use # Set the database for the unwounded data
  
  # We now identify over-expressed ligands/receptors in a cell group and then project gene expression data onto the protein-protein interaction network
  organoid.ccc <- subsetData(organoid.ccc) # We subset the expression data of signalling genes to save on computational cost
  organoid.ccc <- identifyOverExpressedGenes(organoid.ccc) # Identify over-expressed genes (I wonder how much the pre-processing in Seurat has an effect on this)
  organoid.ccc <- identifyOverExpressedInteractions(organoid.ccc) # Identify the over-expressed ligand-receptor interactions, which are determined by an over-expressed ligand OR receptor
  organoid.ccc <- projectData(organoid.ccc, PPI.human) # Other option includes PPI.human We're told that we may have to comment these out.
  
  # We now infer the cell-cell communication network by calculating the communication probabilities
  organoid.ccc <- computeCommunProb(organoid.ccc, raw.use = FALSE, population.size = FALSE)
  organoid.ccc <- filterCommunication(organoid.ccc, min.cells = organoidMinCells) # Filter out the clusters with small numbers
  organoid.ccc <- computeCommunProbPathway(organoid.ccc) # Calculate the probabilities at the signalling level
  organoid.ccc <- aggregateNet(organoid.ccc) # Calculates the aggregated cell-cell communication network by counting the links or summing the communication probabilities
  
  organoid.net <- subsetCommunication(organoid.ccc)
  
  write.csv(organoid.net, file = paste("./output/", sample, "_communications.csv", sep=""))
  
}


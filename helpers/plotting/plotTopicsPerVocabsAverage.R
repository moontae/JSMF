## Clear the memory and the console.
rm(list=ls())
cat("\014")  

## Load the libraries and set the working folder.
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)
setwd("D:\\Analysis\\JSMF\\Plots5\\")
outputBase <- "real_TopicsPerVocabs"

## Prepare basic information about the result data.
datasets <- c("nips", "nytimes", "movies", "songs")
baseVocabs <- c(1250, 7500, 1250, 5000)
method <- "ENN"
opt <- "activeSet"
topics <- c("5", "5", "10", "10", "15", "15", "20", "20", "25", "25", "50", "50", "75", "75", "100", "100", "125", "125", "150", "150")
#focuses <- c(9, 11, 13, 15)
focuses <- c(8, 12, 14, 16)

## Get the number of datasets and the number of different vocabularies.
numDatasets <- length(datasets)
numDiffVocabs <- 8
numFocuses <- length(focuses)

## Set the names.
dataNames <- c("NIPS", "NYTimes", "Movies", "Songs")
dataLabels <- c("NIPS (# documents M=1,348  /  average document length=380.5)", "NYTimes (# documents M=263,325  /  average document length=204.9)", "Movies (# documents M=63,041  /  average document Length=142.8)", "Songs (# documents M=14,653  /  average document length=119.2)")
focusNames <- c(paste("K=", topics[focuses[1]], sep=""), paste("K=", topics[focuses[2]], sep=""), paste("K=", topics[focuses[3]], sep=""), paste("K=", topics[focuses[4]], sep=""))

shift <- function(x){
  c(NA, x[1:length(x)-1])
}


## For each dataset,
metricTables <- list()
stdevTables <- list()
for (i in 1:numDatasets)
{
  # Prepare one dataset.
  dataset <- datasets[i]
  dataName <- dataNames[i]
  metricTables[[i]] <- list()
  stdevTables[[i]] <- list()
  
  # For each vocab,
  for (j in 1:numDiffVocabs)
  {
    # Compute the vocab size.
    vocab <- baseVocabs[i]*j
    
    # Parse the corresponding filenames.
    folder <- paste(dataset, "_N-", vocab, "_", method,  "/", opt, "/", sep="")
    metricFile <- paste(folder, "result_all.metrics", sep="")
    print(metricFile)
    
    # Read the metric and standard deviation tables.
    metricTable <- read.table(metricFile, header=T)
    metricTable$Approximation <- shift(metricTable$Approximation)    
    
    # Compute odd rows (against original) and even rows (against processed).
    numRows  <- nrow(metricTable)      
    oddRows  <- seq(1, numRows, 2)
    evenRows <- seq(2, numRows, 2)

    # Add the topic and vocab column.
    metricTable$Topics <- topics[1:numRows]
    
    # Treat all the inf as missing values.
    tempTable <- do.call(data.frame,lapply(metricTable[evenRows, ], function(x) replace(x, is.infinite(x),NA)))
    metricTable <- colMeans(tempTable[1:5, c(1:21)], na.rm=TRUE)
    metricTable <- data.frame(rbind(metricTable, colMeans(tempTable[5:7, c(1:21)], na.rm=TRUE)))
    metricTable <- data.frame(rbind(metricTable, colMeans(tempTable[7:8, c(1:21)], na.rm=TRUE)))

    # Change the scales of some metric values.
    metricTable[, 1] <- log(metricTable[, 1], base=10)
    metricTable[, 2] <- log(metricTable[, 2], base=10)
    metricTable[, 3] <- log(metricTable[, 3], base=10)
    metricTable[, 4] <- log(metricTable[, 4], base=10)
    metricTable[, 7] <- log(metricTable[, 7], base=10)
    metricTable[, 17] <- metricTable[, 17]/1000
    metricTable[, 20] <- log(metricTable[, 20], base=10)
    metricTable[, 21] <- log(metricTable[, 21], base=10)
    
    colnames(metricTable)[17] <- "AnchorQuality"
    
    metricTables[[i]][[j]] <- metricTable
    metricTables[[i]][[j]]$Topics <- c("K=small", "K=medium", "K=large")
    
    # Add one more column indicating the experiment detail.
    
    metricTables[[i]][[j]]$Vocabs <- vocab/1000
    
  }
}

# Prepare grouping shemes.
colors = hcl(h=seq(15, 375, length=numFocuses+1), l=65, c=100)[1:numFocuses]
colorValues = c(colors)
shapeValues = c(0, 1, 2, 5)
shapeValues = c(0, 1, 2, 5, 6, 7, 8, 9)
lineTypes = c("solid", "solid", "solid", "dashed") # alphabetical order.
lineTypes = c("solid", "solid", "solid", "solid", "solid", "solid", "solid", "solid") # alphabetical order.

## For each dataset,
figures <- list()
plots <- list()
for (i in 1:numDatasets)
{
  ## Prepare one dataset.
  dataset <- datasets[i]
  dataName <- dataNames[i]
  dataLabel <- dataLabels[i]
  
  ## Vertically combine multiple data in a row for each dataset
  figures[[i]] <- metricTables[[i]][[1]]
  for (j in 2:numDiffVocabs) {
    figures[[i]] <- rbind(figures[[i]], metricTables[[i]][[j]])   
  }
  
  ## change the wide table where each column corresponds to a metric to a narrow table where 
  ## the first two columns are [Vocab, Topics] (primary key) and 
  ## two new columns are [Metric, value] (Metric correspond to the original column header, value is the corresponding value)
  figures[[i]] <- melt(figures[[i]], id.vars=c("Vocabs", "Topics"), variable.name="Metric")  
  
  ## create a ggplot object with using only certain subset of metrics
  ## aes function defines X-axis="Vocabs" / Y-axis="value" with differentiating color by "Algorithm"
  plots[[i]] <- ggplot(subset(figures[[i]], Metric %in% c("Recovery", "Approximation", "BaseRecovery", "AnchorQuality", "Entropy", "Sparsity", "RectifyTime", "FactorizeTime")), aes(Vocabs, value, color=Topics, shape=Topics, linetype=Topics))
  
  ## display the plot, plus a "line" geometry and a faceting by metric.
  plots[[i]] <- plots[[i]] + scale_linetype_manual(values=lineTypes) + scale_color_manual(values=colorValues) + scale_shape_manual(values=shapeValues) + geom_point(size=2) + geom_line(alpha=0.75) + facet_wrap(~ Metric, scales="free", nrow=1) + labs(title=dataLabel) + theme(plot.margin=unit(c(0.1, 0, 0.1, 0), "in"), plot.title=element_text(size=10, hjust=0.5), axis.title.x=element_blank(), axis.title.y=element_blank())
  #plots[[i]] <- plots[[i]] + scale_linetype_manual(values=lineTypes) + scale_color_manual(values=colorValues) + scale_shape_manual(values=shapeValues) + geom_point(size=2) + geom_line(alpha=0.75) + facet_wrap(~ Metric, scales="free", nrow=1) + labs(title=dataLabel) + theme(plot.margin=unit(c(0.1, 0, 0.1, 0), "in"), plot.title=element_text(size=10, hjust=0.5), axis.title.x=element_blank(), axis.title.y=element_blank())
  #plots[[i]] <- plots[[i]] + scale_linetype_manual(values=lineTypes) + scale_color_manual(values=colorValues) + scale_shape_manual(values=shapeValues) + geom_point(size=2) + geom_line(alpha=0.75) + facet_wrap(~ Metric, scales="free", nrow=1) + labs(title=dataLabel) + theme(plot.margin=unit(c(0.1, 0, 0.1, 0), "in"), plot.title=element_text(size=10, hjust=0.5), axis.title.x=element_blank(), axis.title.y=element_blank()) + scale_x_log10(breaks=c(5, 10, 15, 25, 50, 100, 150), minor_breaks=c(20, 75, 125))
  
  
  ## save the plot 
  outputFile <- paste(outputBase, "-", dataName, ".pdf", sep="")
  ggsave(outputFile, height=2.3, width=15)  
}  
  
pdf(paste(outputBase, ".pdf", sep=""), height=(2.1*numDatasets), width=15)
grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], ncol=1)
#grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], plots[[5]], ncol=1)
dev.off()




## clear the memory and console
rm(list=ls())
cat("\014")  

## load the libraries and set the working folder
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)
setwd("D:\\Plots3\\")
outputBase <- "real_VocabsPerTopics"

## prepare basic information about the result data
datasets <- c("nips", "nytimes", "movies", "songs")
vocabs <- c(1250, 7500, 1250, 5000)
method <- "ENN"
opt <- "activeSet"
topics <- c(5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 50, 50, 75, 75, 100, 100, 125, 125, 150, 150)

## get the length of each set
numDatasets <- length(datasets)
numVocabs <- length(vocabs)

## setup the name of methods
dataNames <- c("NIPS", "NYTimes", "Movies", "Songs")
dataLabels <- c("NIPS (# documents M=1,348  /  base vocabulary N=1.25k /  average document length=380.5)", "NYTimes (# documents M=263,325  /  base vocabulary N=7.5k  /  average document length=204.9)", "Movies (# documents M=63,041  /  base vocabulary N=1.25k  /  average document Length=142.8)", "Songs (# documents M=14,653  /  base vocabulary N=5k  /  average document length=119.2)")
vocabNames <- c("1x", "2x", "4x", "8x")
numVocabNames <- length(vocabNames)

## for each dataset
metricTables <- list()
stdevTables <- list()
for (i in 1:numDatasets)
{
  # prepare one dataset
  dataset <- datasets[i]
  dataName <- dataNames[i]
  metricTables[[i]] <- list()
  stdevTables[[i]] <- list()
  
  # for each vocab
  for (j in 1:numVocabNames)
  {
    # prepare one vocab
    vocab <- vocabs[i]
    vocabName <- vocabNames[j]
    print(vocab)
    
    # parse the corresponding filenames
    folder <- paste(dataset, "_N-", vocab*(2^(j-1)), "_", method,  "/", opt, "/", sep="")
    metricFile <- paste(folder, "result_all.metrics", sep="")
    print(metricFile)
    
    # read the value and standard deviation tables
    metricTable <- read.table(metricFile, header=T)
    metricTable[, 1] <- log(metricTable[, 1], base=10)
    metricTable[, 2] <- log(metricTable[, 2], base=10)
    metricTable[, 3] <- log(metricTable[, 3], base=10)
    metricTable[, 4] <- log(metricTable[, 4], base=10)
    metricTable[, 17] <- log(metricTable[, 19], base=10)
    metricTable[, 18] <- log(metricTable[, 20], base=10)
    
    # compute odd rows (against original) and even rows (against processed)
    numRows  <- nrow(metricTable)      
    oddRows  <- seq(1, numRows, 2)
    evenRows <- seq(2, numRows, 2)
    
    # add the topic columns
    metricTable$Topics <- topics[1:numRows]
    
    # split the table and save separately
    metricTables[[i]][[j]] <- metricTable[oddRows, ]
    
    ## add one more column indicating the experiment detail
    metricTables[[i]][[j]]$Category <- vocabName
  }
}

# start melting
colors = hcl(h=seq(15, 375, length=numVocabs+1), l=65, c=100)[1:numVocabs]
colorValues = c(colors)
shapeValues = c(0, 1, 2, 5)
#shapeValues = c(0, 1, 2, 5, 6)
lineTypes = c("solid", "solid", "solid", "solid") # alphabetical order of the methodNames.


## for each mode
figures <- list()
plots <- list()
for (i in 1:numDatasets)
{
  ## prepare one dataset
  dataset <- datasets[i]
  dataName <- dataNames[i]
  dataLabel <- dataLabels[i]
  
  ## vertically combine multiple data in a row for each dataset
  table <- metricTables[[i]][[1]]
  figures[[i]] <- table[, colnames(table) != 'CondNumber']
  for (j in 2:numVocabs)
    figures[[i]] <- rbind(figures[[i]], metricTables[[i]][[j]])   
  
  ## change the wide table where each column corresponds to a metric to a narrow table where 
  ## the first two columns are [Topics, Category] (primary key) and 
  ## two new columns are [Metric, value] (Metric correspond to the original column header, value is the corresponding value)
  figures[[i]] <- melt(figures[[i]], id.vars=c("Topics", "Category"), variable.name="Metric")  
  
  ## create a ggplot object with using only certain subset of metrics
  ## aes function defines X-axis="Topics" / Y-axis="value" with differentiating color by "Algorithm"
  plots[[i]] <- ggplot(subset(figures[[i]], Metric %in% c("Recovery", "Approximation", "Dominancy", "Specificity", "Dissimilarity", "RectifyTime", "FactorizeTime")), aes(Topics, value, color=Category, shape=Category, linetype=Category))
  maxTopics = max(figures[[i]]$Topics)                
  
  ## display the plot, plus a "line" geometry and a faceting by metric.
  plots[[i]] <- plots[[i]] + scale_linetype_manual(values=lineTypes) + scale_color_manual(values=colorValues) + scale_shape_manual(values=shapeValues) + geom_point(size=2) + geom_line(alpha=0.75) + facet_wrap(~ Metric, scales="free", nrow=1) + labs(title=dataLabel) + theme(plot.margin=unit(c(0.1, 0, 0.1, 0), "in"), plot.title=element_text(size=10, hjust=0.5), axis.title.x=element_blank(), axis.title.y=element_blank()) + scale_x_log10(breaks=c(5, 10, 15, 25, 50, 100, 150), minor_breaks=c(20, 75, 125))
  
  ## save the plot 
  outputFile <- paste(outputBase, "-", dataName, ".pdf", sep="")
  ggsave(outputFile, height=2.3, width=15)  
}  
  
pdf(paste(outputBase, ".pdf", sep=""), height=(2.1*numDatasets), width=15)
grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], ncol=1)
#grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], plots[[5]], ncol=1)
dev.off()




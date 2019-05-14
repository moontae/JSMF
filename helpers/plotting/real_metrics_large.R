## clear the memory and console
rm(list=ls())
cat("\014")  

## load the libraries and set the working folder
library(ggplot2)
library(reshape2)
library(gridExtra)
library(grid)
setwd("D:\\Plots\\")

## prepare basic information about the result data
datasets <- c("nytimes_N-7500", "nytimes_N-15000", "nytimes_N-30000", "nytimes_N-60000")
methods <- c("NONE", "ENN")
opt <- "activeSet"
topics <- c(5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 50, 50, 75, 75, 100, 100, 125, 125, 150, 150, 200, 200, 250, 250, 300, 300)

## get the length of each set
numDatasets <- length(datasets)
numMethods <- length(methods)

## setup the name of methods
dataNames <- c("NYTimes", "NYTimes", "NYTimes", "NYTimes")
dataLabels <- c("NYTimes (# documents M=263,325  /  # vocabulary N=7.5k  /  average document length=204.9)", "NYTimes (# documents M=263,325  /  # vocabulary N=15k  /  average document length=204.9)", "NYTimes (# documents M=263,325  /  # vocabulary N=30k  /  average document length=204.9)", "NYTimes (# documents M=263,325  /  # vocabulary N=60k  /  average document length=204.9)")
methodNames <- c("NONE", "ENN")

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
  
  # for each method
  for (j in 1:numMethods)
  {
    # prepare one method
    method <- methods[j]
    methodName <- methodNames[j]
    print(method)
    
    # parse the corresponding filenames
    folder <- paste(dataset, "_", method, "/", opt, "/", sep="")
    metricFile <- paste(folder, "result_all.metrics", sep="")
    print(metricFile)
    
    # read the value and standard deviation tables
    metricTable <- read.table(metricFile, header=T)
    metricTable[, 1] <- log(metricTable[, 1], base=1.8)
    metricTable[, 4] <- log(metricTable[, 4], base=1.8)
    
    colnames(metricTable)[10]  <- "Soft-Dissimilarity"
    colnames(metricTable)[13]  <- "Soft-BasisRank"
    colnames(metricTable)[15]  <- "Soft-BasisQuality"

    # compute odd rows (against original) and even rows (against processed)
    numRows  <- nrow(metricTable)      
    oddRows  <- seq(1, numRows, 2)
    evenRows <- seq(2, numRows, 2)
    
    # add the topic columns
    metricTable$Topics <- topics[1:numRows]
    
    # split the table and save separately
    metricTables[[i]][[j]] <- metricTable[oddRows, ]
    
    ## add one more column indicating the experiment detail
    metricTables[[i]][[j]]$Category <- methodName
  }
}

# start melting
colors = hcl(h=seq(15, 375, length=numMethods+1), l=65, c=100)[1:numMethods]
colorValues = c(colors)
shapeValues = c(0, 1, 2, 5)
#shapeValues = c(0, 1, 2, 5, 6)
lineTypes = c("solid", "solid", "solid", "solid")


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
  for (j in 2:numMethods)
    figures[[i]] <- rbind(figures[[i]], metricTables[[i]][[j]])   
  
  ## change the wide table where each column corresponds to a metric to a narrow table where 
  ## the first two columns are [Topics, Category] (primary key) and 
  ## two new columns are [Metric, value] (Metric correspond to the original column header, value is the corresponding value)
  figures[[i]] <- melt(figures[[i]], id.vars=c("Topics", "Category"), variable.name="Metric")  
  
  ## create a ggplot object with using only certain subset of metrics
  ## aes function defines X-axis="Topics" / Y-axis="value" with differentiating color by "Algorithm"
  plots[[i]] <- ggplot(subset(figures[[i]], Metric %in% c("Recovery", "Legalty", "Approximation", "Dominancy", "Entropy", "Specificity", "Dissimilarity", "Coherence", "Sparsity")), aes(Topics, value, color=Category, shape=Category, linetype=Category))
  maxTopics = max(figures[[i]]$Topics)                
  
  ## display the plot, plus a "line" geometry and a faceting by metric.
  plots[[i]] <- plots[[i]] + scale_linetype_manual(values=lineTypes) + scale_color_manual(values=colorValues) + scale_shape_manual(values=shapeValues) + geom_point(size=2) + geom_line(alpha=0.75) + facet_wrap(~ Metric, scales="free", nrow=1) + labs(title=dataLabel) + theme(plot.margin=unit(c(0.1, 0, 0.1, 0), "in"), plot.title=element_text(size=10, hjust=0.5), axis.title.x=element_blank(), axis.title.y=element_blank()) + scale_x_log10(breaks=c(5, 10, 15, 25, 50, 100, 150, 200, 300), minor_breaks=c(20, 75, 125, 250))
  
  ## save the plot 
  outputFile <- paste("real_metrics_large", "-", dataName, ".pdf", sep="")
  ggsave(outputFile, height=2.3, width=15)  
}  
  
pdf(paste("real_metrics_large", ".pdf", sep=""), height=(2.1*numDatasets), width=15)
grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], ncol=1)
#grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], plots[[5]], ncol=1)
dev.off()




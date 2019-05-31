#!/usr/bin/Rscript

library(scales)
library(mixtools)

data <- read.table('data', header=FALSE)

data

plot(data$V2, data$V3, pch=20, cex=.25, col=factor(data$V1))

mu <- c(162.472, 198.738)
sigma <- matrix(c(575.282,341.73,341.73,352.372), 2, 2)
ellipse(mu, sigma, npoints = 200, col="black", newplot = FALSE)

mu1 <- c(105.722, 136.834)
sigma1 <- matrix(c(528.233,371.895,371.895,407.627), 2, 2)
ellipse(mu1, sigma1, npoints = 200, col="red", newplot = FALSE)

#3
#covariance for cluster 0 [[575.282,341.731],[341.731,352.372]]
#mean for cluster 0 [162.472, 198.738, 207.384]
#covariance for cluster 1 [[528.233,371.895],[371.895,407.627]]
#mean for cluster 1 [105.722, 136.834, 139.554]

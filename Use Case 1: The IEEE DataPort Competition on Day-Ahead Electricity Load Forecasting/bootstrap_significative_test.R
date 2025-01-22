library(opera)
WeakL <- read.csv2("data/WeakL_mae99.csv", sep=",", stringsAsFactors = F)
WeakL$Load <- as.numeric(WeakL$Load)
WeakL$WeakL <-as.numeric(WeakL$WeakL)
mean(abs(tail(WeakL$Load-WeakL$WeakL,720)))

prev_corr <- readRDS('data/experts_corr.RDS')
test3 <- which(prev_corr$Time >= as.POSIXct(strptime("2020-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))
prev_corr <- prev_corr[test3,]

opera_discriminate <- function(experts,y,hour) {
  yhat <- numeric(length(y))
  for (h in 0:23) {
    sel <- which(hour==h)
    agg <- opera::mixture(Y=y[sel], experts=experts[sel,], model="MLpol", loss.gradient=T,
                          loss.type='absolute')
    yhat[sel] <- rowSums((if (h < 8) agg$weights else agg$weights[c(1,1:(length(sel)-1)),]) * experts[sel,])
  }
  yhat
}
I3E_win <- opera_discriminate(data.matrix(prev_corr[,4:31]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load,720)-tail(I3E_win, 720)))



Y1 <- abs(tail(prev_corr$Load,720)-tail(WeakL$WeakL,720))
Y2 <- abs(tail(prev_corr$Load,720)-tail(I3E_win, 720))
skill_score <- 1-mean(Y1)/mean(Y2)

## Variance bootstrap estimation
MC <- 3000 #number of monte carlo run
n <- 720
l <- 24 #block size
N <- n-l+1
set.seed(988)
MAE1 <- rep(0,MC)
for(i in 1:MC){
  MAE1[i] <- mean(sapply(sample(x=c(1:N),size=n/l), FUN = function(j){
    return(Y1[j:(j+l-1)])
  }))
}
set.seed(988)
MAE2 <- rep(0,MC)
for(i in 1:MC){
  MAE2[i] <- mean(sapply(sample(x=c(1:N),size=n/l), FUN = function(j){
    return(Y2[j:(j+l-1)])
  }))
}


skill_sample <- 1-MAE1/MAE2
skill_sample_sd <- sd(skill_sample)
CI_low <- skill_score+qnorm(p = 0.10)*skill_sample_sd #- 0.007 >0 !


library(opera)
PIKL_expert_off <- read.csv2("PIKL_expert_corr.csv", sep=",", stringsAsFactors = F)
PIKL_expert_off$PIKL <- as.numeric(PIKL_expert_off$PIKL)

PIKL_expert_on <- read.csv2("PIKL_expert_corr_online.csv", sep=",", stringsAsFactors = F)
PIKL_expert_on$PIKL <- as.numeric(PIKL_expert_on$PIKL)

prev_corr <- readRDS('experts_corr.RDS')
prev_corr$PIKL_off <- PIKL_expert_off$PIKL
prev_corr$PIKL_on <- PIKL_expert_on$PIKL
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


# 7 AR
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,4:10]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))
# 7 Lin
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,11:17]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))


# 7 GAM
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,18:24]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))
# 7 MLP
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,25:31]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))

# 28 experts
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,4:31]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))

# All experts
yhat.opera.discr <- opera_discriminate(data.matrix(cbind(prev_corr[,4:33])), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))

# All experts without GAM
yhat.opera.discr <- opera_discriminate(data.matrix(prev_corr[,-c(1:3,18:24)]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-yhat.opera.discr,720)))



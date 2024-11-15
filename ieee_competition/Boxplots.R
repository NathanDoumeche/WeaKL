library(opera)
library(boot)
library(tseries)
library(ggplot2)

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

# Import experts
pikl_expert_files <- c("PIKL_expert_corr.csv", "PIKL_expert_corr_online.csv", "PIKL_expert_corr_99.csv", "PIKL_expert_corr_online_99.csv")
pikl_expert_names <- c("Offline", "Online", "Offline 99", "Online 99")

pikl_experts.df <- sapply(pikl_expert_files, FUN = function(filename){
  as.numeric(read.csv2(paste0("data/",filename), header = T, sep = ",", stringsAsFactors = F)$PIKL)
})
colnames(pikl_experts.df) <- pikl_expert_names

prev_corr <- readRDS('data/experts_corr.RDS')
prev_corr <- cbind(prev_corr, pikl_experts.df)
test3 <- which(prev_corr$Time >= as.POSIXct(strptime("2020-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"), tz="UTC"))
prev_corr <- prev_corr[test3,]

I3E_win <- opera_discriminate(data.matrix(prev_corr[,4:31]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-I3E_win,720)))

agg_pikl <- opera_discriminate(data.matrix(prev_corr[,-c(1:3,18:24,34,35)]), prev_corr$Load, prev_corr$Hour)
mean(abs(tail(prev_corr$Load-agg_pikl,720)))

mean(abs(tail(prev_corr$Load - prev_corr$`Online 99`, 720)))


# MAE skill score block bootstrap
set.seed(948)
I3E_win_bs <- tsbootstrap(tail(prev_corr$Load,720)-tail(I3E_win,720), nb=500, b=24, type=c("block"))

set.seed(948)
agg_pikl_bs <- tsbootstrap(tail(prev_corr$Load,720)-tail(agg_pikl,720), nb=500, b=24, type=c("block"))

set.seed(948)
pikl_bs <- tsbootstrap(tail(prev_corr$Load,720)-tail(prev_corr$`Online 9.9`,720), nb=500, b=24, type=c("block"))

agg_pikl_score <- 1 - (apply(agg_pikl_bs, 2, FUN = function(x){mean(abs(x))}) / apply(I3E_win_bs, 2, FUN = function(x){mean(abs(x))}))
pikl_score <- 1 - (apply(pikl_bs, 2, FUN = function(x){mean(abs(x))}) / apply(I3E_win_bs, 2, FUN = function(x){mean(abs(x))}))

ggplot(data = data.frame(Team = rep(c("Aggregation with PIKL", "PIKL 99"), each = 500), Score=c(agg_pikl_score,pikl_score)), aes(x=Team, y=Score, fill=Team))+
  geom_boxplot(outlier.shape = NA, show.legend = F)+labs(x="", y="")+
  theme_light()

Q = seq(0,0.25, length.out=1000)
ggplot(data = data.frame(Quantiles = Q, Agg = quantile(agg_pikl_score, Q), PIKL = quantile(pikl_score, Q)))+
  geom_line(mapping = aes(x= Quantiles, y= Agg, color= "Aggregation")) + geom_line(mapping = aes(x = Quantiles, y = PIKL, color = "PIKL 99"))+
  geom_hline(yintercept = 0, color="black", linetype = "dashed")+
  scale_color_manual(values = c("Aggregation" = "firebrick3", "PIKL 99" = "darkmagenta"))+labs(x="level", y="q")+labs(color="", title="Empirical quantiles of MAE skill scores")+theme_light()

# NonOverlapping bootstrap

Reference <- readRDS("data/data.RDS")
Reference <- Reference[,c(1,2)]
Reference$Time <- as.POSIXct(Reference$Time, tz="UTC")

Reference$forecast <- numeric(length(Reference[,1]))
Reference$forecast[which(format(Reference$Time, "%w")==0)]=c(rep(NA,24),head(Reference$Load[which(format(Reference$Time, "%w")==0)],-24))
Reference$forecast[which(format(Reference$Time, "%w")==6)]=c(rep(NA,24),head(Reference$Load[which(format(Reference$Time, "%w")==6)],-24))
Reference$forecast[which(format(Reference$Time, "%w")%in%1:5)]=c(rep(NA,48),head(Reference$Load[which(format(Reference$Time, "%w")%in%1:5)],-48))
Reference$forecast_MON =numeric(length(Reference$forecast))
Reference$forecast_MON[which(format(Reference$Time, "%w")%in%1:5)]=c(rep(NA,24),head(Reference$Load[which(format(Reference$Time, "%w")%in%1:5)],-24))
Reference$forecast[which(format(Reference$Time, "%w")==1)]=Reference$forecast_MON[which(format(Reference$Time, "%w")==1)]

mean(abs(tail(prev_corr$Load, 720)-tail(Reference$forecast,720)))


daily_mae <- function(vector){
  as.numeric(sapply(split(vector, (seq_along(vector) - 1) %/% 24), function(x){mean(abs(x))}))
}

df_leaderboard <- read.csv2(file ="R/LeaderBoard_Final.csv", sep=",", stringsAsFactors = F )
df_leaderboard <- df_leaderboard[,-c(ncol(df_leaderboard),ncol(df_leaderboard)-1)]
df_leaderboard <- rbind(c("Agg_pikl", daily_mae(tail(prev_corr$Load,720) - tail(agg_pikl, 720))), df_leaderboard)
df_leaderboard <- rbind(c("pikl_99", daily_mae(tail(prev_corr$Load,720) - tail(prev_corr$`Online 9.9`, 720))), df_leaderboard) 

Teams <- df_leaderboard$Team[1:12]
set.seed(948)
ref_bs <- as.numeric(boot(as.numeric(df_leaderboard[grep("Benchmark", df_leaderboard$Team),-c(1)]), statistic = function(data, indices){mean(abs(data[indices]))},R=N_samples)$t)
team_score_bs <- as.data.frame(sapply(Teams, FUN = function(team){
  set.seed(948)
  res <-as.numeric(boot(as.numeric(df_leaderboard[df_leaderboard$Team==team,-c(1)]), statistic = function(data, indices){mean(abs(data[indices]))},R=500)$t)
  1- res/ref_bs
}, simplify=T))

colnames(team_score_bs) <- Teams

team_score_bs <- reshape2::melt(team_score_bs, variable.name = "Team", value.name = "score")

ggplot(team_score_bs, aes(x=Team, y=score, fill=Team))+geom_boxplot(show.legend = F, outlier.shape = NA)+labs(x="", y="")+theme_light()+
  theme(text=element_text(family="serif"),axis.text.x = element_text(angle = -45))

  

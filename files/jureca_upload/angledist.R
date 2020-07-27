library(ggplot2)
library(latex2exp)
library(diptest)
library(modeest)
setwd("~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload")

path=read.csv("path.csv",header = TRUE,sep=',')$path

b_name <- function(b){
  split_b = strsplit(b, "")[[1]]
  b1 = split_b[1]
  b3 = split_b[3]
  
  #b_string = paste(b1,"_",b3,sep = "")

  
  b_string = paste(b1,"_",b3,sep = "")
  return(b_string)
}

b_list = 2 * read.csv(paste(path,"b_list.csv",sep = ""),header = TRUE, sep = ",")$b
N_list = read.csv(paste(path,"N_ped_list.csv",sep = ""),header = TRUE, sep = ",")$N

D = c()
p = c()

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (b in b_list){
  #print(as.numeric(b * 1.0001))
  b_string = b_name(toString(b * 1.0000001))
  data = read.csv(paste(path,"distributions/","dist_",b_string,".csv",sep = ""),header = TRUE,sep=',')
  #print(head(data))
  #angle = data$angle
  angle = data[data$angle > 0.0, ]$angle
  plot = ggplot(data, aes(x=angle)) + geom_histogram(aes(y = stat(count / sum(count))))  + theme_classic( base_size = 14)   + ylab(TeX("$\\rho_0$")) + xlab(TeX("$\\Theta$"))
  print(plot)
  ggsave(paste(path,"/plots/angledist/","dist_",b_string,".png",sep=""),width = 5, height = 4)
  #h = hist(angle)
  #print(h$counts)
  a = dip.test(angle)
  #print(a)
  #plot(density(angle),main=b_string)
  print(mfv(angle))
  print("uh")
  #print(a$p.value)
  #D = c(D,a$D)
  #print(a[[2]])
  p = c(p,a$p.value)
  D = c(D,a$statistic)
}

#dfp = data.frame(b = b_list, p = p, D = D)
#print(dfp)
#print(ggplot(dfp, aes(b,p)) + geom_point())
#print(ggplot(dfp, aes(b,D)) + geom_point())

#dfD = data.frame(b=b,D=D)
#print(dfp)

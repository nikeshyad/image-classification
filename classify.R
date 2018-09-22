source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")
list.of.packages <- c("imager", "gridExtra","ggplot2","plyr","neuralnet","e1071","randomForest")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

install.packages("class")
install.packages("gmodels")
library(imager)
library(EBImage)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(randomForest)
library(e1071)
library(neuralnet)
library(class) 
library(gmodels)

gray_pixels <- function(Image) {
  im <- load.image(Image)
  gray <- grayscale(im)
  resize.im <- resize(gray, w=200, h=200)
  mat <- matrix(resize.im, nrow=1)
  return(mat)
}

#Load Car Images
car1 <- gray_pixels("Car/1.jpg")
car2 <- gray_pixels("Car/2.jpg")
car3 <- gray_pixels("Car/3.jpg")
car4 <- gray_pixels("Car/4.jpg")
car5 <- gray_pixels("Car/5.jpg")
car6 <- gray_pixels("Car/6.jpg")
car7 <- gray_pixels("Car/7.jpg")
car8 <- gray_pixels("Car/8.jpg")
car9 <- gray_pixels("Car/9.jpg")
car10 <- gray_pixels("Car/10.jpg")
car11 <- gray_pixels("Car/11.jpg")
car12 <- gray_pixels("Car/12.jpg")
car13 <- gray_pixels("Car/13.jpg")
car14 <- gray_pixels("Car/14.jpg")
car15 <- gray_pixels("Car/15.jpg")
car16 <- gray_pixels("Car/16.jpg")
car17 <- gray_pixels("Car/17.jpg")
car18 <- gray_pixels("Car/18.jpg")
car19 <- gray_pixels("Car/19.jpg")
car20 <- gray_pixels("Car/20.jpg")
car21 <- gray_pixels("Car/21.jpg")
car22 <- gray_pixels("Car/22.jpg")
car23 <- gray_pixels("Car/23.jpg")
car24 <- gray_pixels("Car/24.jpg")
car25 <- gray_pixels("Car/25.jpg")
car26 <- gray_pixels("Car/26.jpg")
car27 <- gray_pixels("Car/27.jpg")
car28 <- gray_pixels("Car/28.jpg")
car29 <- gray_pixels("Car/29.jpg")
car30 <- gray_pixels("Car/30.jpg")
car31 <- gray_pixels("Car/31.jpg")
car32 <- gray_pixels("Car/32.jpg")
car33 <- gray_pixels("Car/33.jpg")
car34 <- gray_pixels("Car/34.jpg")
car35 <- gray_pixels("Car/35.jpg")
car36 <- gray_pixels("Car/36.jpg")
car37 <- gray_pixels("Car/37.jpg")
car38 <- gray_pixels("Car/38.jpg")
car39 <- gray_pixels("Car/39.jpg")
car40 <- gray_pixels("Car/40.jpg")

#Load Tree Images
tree1 <- gray_pixels("Tree/1.jpg")
tree2 <- gray_pixels("Tree/2.jpg")
tree3 <- gray_pixels("Tree/3.jpg")
tree4 <- gray_pixels("Tree/4.jpg")
tree5 <- gray_pixels("Tree/5.jpg")
tree6 <- gray_pixels("Tree/6.jpg")
tree7 <- gray_pixels("Tree/7.jpg")
tree8 <- gray_pixels("Tree/8.jpg")
tree9 <- gray_pixels("Tree/9.jpg")
tree10 <- gray_pixels("Tree/10.jpg")
tree11 <- gray_pixels("Tree/11.jpg")
tree12 <- gray_pixels("Tree/12.jpg")
tree13 <- gray_pixels("Tree/13.jpg")
tree14 <- gray_pixels("Tree/14.jpg")
tree15 <- gray_pixels("Tree/15.jpg")
tree16 <- gray_pixels("Tree/16.jpg")
tree17 <- gray_pixels("Tree/17.jpg")
tree18 <- gray_pixels("Tree/18.jpg")
tree19 <- gray_pixels("Tree/19.jpg")
tree20 <- gray_pixels("Tree/20.jpg")
tree21 <- gray_pixels("Tree/21.jpg")
tree22 <- gray_pixels("Tree/22.jpg")
tree23 <- gray_pixels("Tree/23.jpg")
tree24 <- gray_pixels("Tree/24.jpg")
tree25 <- gray_pixels("Tree/25.jpg")
tree26 <- gray_pixels("Tree/26.jpg")
tree27 <- gray_pixels("Tree/27.jpg")
tree28 <- gray_pixels("Tree/28.jpg")
tree29 <- gray_pixels("Tree/29.jpg")
tree30 <- gray_pixels("Tree/30.jpg")
tree31 <- gray_pixels("Tree/31.jpg")
tree32 <- gray_pixels("Tree/32.jpg")
tree33 <- gray_pixels("Tree/33.jpg")
tree34 <- gray_pixels("Tree/34.jpg")
tree35 <- gray_pixels("Tree/35.jpg")
tree36 <- gray_pixels("Tree/36.jpg")
tree37 <- gray_pixels("Tree/37.jpg")
tree38 <- gray_pixels("Tree/38.jpg")
tree39 <- gray_pixels("Tree/39.jpg")
tree40 <- gray_pixels("Tree/40.jpg")

#Load Waterfall Images
waterfall1 <- gray_pixels("Waterfall/1.jpg")
waterfall2 <- gray_pixels("Waterfall/2.jpg")
waterfall3 <- gray_pixels("Waterfall/3.jpg")
waterfall4 <- gray_pixels("Waterfall/4.jpg")
waterfall5 <- gray_pixels("Waterfall/5.jpg")
waterfall6 <- gray_pixels("Waterfall/6.jpg")
waterfall7 <- gray_pixels("Waterfall/7.jpg")
waterfall8 <- gray_pixels("Waterfall/8.jpg")
waterfall9 <- gray_pixels("Waterfall/9.jpg")
waterfall10 <- gray_pixels("Waterfall/10.jpg")
waterfall11 <- gray_pixels("Waterfall/11.jpg")
waterfall12 <- gray_pixels("Waterfall/12.jpg")
waterfall13 <- gray_pixels("Waterfall/13.jpg")
waterfall14 <- gray_pixels("Waterfall/14.jpg")
waterfall15 <- gray_pixels("Waterfall/15.jpg")
waterfall16 <- gray_pixels("Waterfall/16.jpg")
waterfall17 <- gray_pixels("Waterfall/17.jpg")
waterfall18 <- gray_pixels("Waterfall/18.jpg")
waterfall19 <- gray_pixels("Waterfall/19.jpg")
waterfall20 <- gray_pixels("Waterfall/20.jpg")
waterfall21 <- gray_pixels("Waterfall/21.jpg")
waterfall22 <- gray_pixels("Waterfall/22.jpg")
waterfall23 <- gray_pixels("Waterfall/23.jpg")
waterfall24 <- gray_pixels("Waterfall/24.jpg")
waterfall25 <- gray_pixels("Waterfall/25.jpg")
waterfall26 <- gray_pixels("Waterfall/26.jpg")
waterfall27 <- gray_pixels("Waterfall/27.jpg")
waterfall28 <- gray_pixels("Waterfall/28.jpg")
waterfall29 <- gray_pixels("Waterfall/29.jpg")
waterfall30 <- gray_pixels("Waterfall/30.jpg")
waterfall31 <- gray_pixels("Waterfall/31.jpg")
waterfall32 <- gray_pixels("Waterfall/32.jpg")
waterfall33 <- gray_pixels("Waterfall/33.jpg")
waterfall34 <- gray_pixels("Waterfall/34.jpg")
waterfall35 <- gray_pixels("Waterfall/35.jpg")
waterfall36 <- gray_pixels("Waterfall/36.jpg")
waterfall37 <- gray_pixels("Waterfall/37.jpg")
waterfall38 <- gray_pixels("Waterfall/38.jpg")
waterfall39 <- gray_pixels("Waterfall/39.jpg")
waterfall40 <- gray_pixels("Waterfall/40.jpg")

#Load Beach Images
beach1 <- gray_pixels("Beach/1.jpg")
beach2 <- gray_pixels("Beach/2.jpg")
beach3 <- gray_pixels("Beach/3.jpg")
beach4 <- gray_pixels("Beach/4.jpg")
beach5 <- gray_pixels("Beach/5.jpg")
beach6 <- gray_pixels("Beach/6.jpg")
beach7 <- gray_pixels("Beach/7.jpg")
beach8 <- gray_pixels("Beach/8.jpg")
beach9 <- gray_pixels("Beach/9.jpg")
beach10 <- gray_pixels("Beach/10.jpg")
beach11 <- gray_pixels("Beach/11.jpg")
beach12 <- gray_pixels("Beach/12.jpg")
beach13 <- gray_pixels("Beach/13.jpg")
beach14 <- gray_pixels("Beach/14.jpg")
beach15 <- gray_pixels("Beach/15.jpg")
beach16 <- gray_pixels("Beach/16.jpg")
beach17 <- gray_pixels("Beach/17.jpg")
beach18 <- gray_pixels("Beach/18.jpg")
beach19 <- gray_pixels("Beach/19.jpg")
beach20 <- gray_pixels("Beach/20.jpg")
beach21 <- gray_pixels("Beach/21.jpg")
beach22 <- gray_pixels("Beach/22.jpg")
beach23 <- gray_pixels("Beach/23.jpg")
beach24 <- gray_pixels("Beach/24.jpg")
beach25 <- gray_pixels("Beach/25.jpg")
beach26 <- gray_pixels("Beach/26.jpg")
beach27 <- gray_pixels("Beach/27.jpg")
beach28 <- gray_pixels("Beach/28.jpg")
beach29 <- gray_pixels("Beach/29.jpg")
beach30 <- gray_pixels("Beach/30.jpg")
beach31 <- gray_pixels("Beach/31.jpg")
beach32 <- gray_pixels("Beach/32.jpg")
beach33 <- gray_pixels("Beach/33.jpg")
beach34 <- gray_pixels("Beach/34.jpg")
beach35 <- gray_pixels("Beach/35.jpg")
beach36 <- gray_pixels("Beach/36.jpg")
beach37 <- gray_pixels("Beach/37.jpg")
beach38 <- gray_pixels("Beach/38.jpg")
beach39 <- gray_pixels("Beach/39.jpg")
beach40 <- gray_pixels("Beach/40.jpg")

#Load Mountain Images
mountain1 <- gray_pixels("Mountain/1.jpg")
mountain2 <- gray_pixels("Mountain/2.jpg")
mountain3 <- gray_pixels("Mountain/3.jpg")
mountain4 <- gray_pixels("Mountain/4.jpg")
mountain5 <- gray_pixels("Mountain/5.jpg")
mountain6 <- gray_pixels("Mountain/6.jpg")
mountain7 <- gray_pixels("Mountain/7.jpg")
mountain8 <- gray_pixels("Mountain/8.jpg")
mountain9 <- gray_pixels("Mountain/9.jpg")
mountain10 <- gray_pixels("Mountain/10.jpg")
mountain11 <- gray_pixels("Mountain/11.jpg")
mountain12 <- gray_pixels("Mountain/12.jpg")
mountain13 <- gray_pixels("Mountain/13.jpg")
mountain14 <- gray_pixels("Mountain/14.jpg")
mountain15 <- gray_pixels("Mountain/15.jpg")
mountain16 <- gray_pixels("Mountain/16.jpg")
mountain17 <- gray_pixels("Mountain/17.jpg")
mountain18 <- gray_pixels("Mountain/18.jpg")
mountain19 <- gray_pixels("Mountain/19.jpg")
mountain20 <- gray_pixels("Mountain/20.jpg")
mountain21 <- gray_pixels("Mountain/21.jpg")
mountain22 <- gray_pixels("Mountain/22.jpg")
mountain23 <- gray_pixels("Mountain/23.jpg")
mountain24 <- gray_pixels("Mountain/24.jpg")
mountain25 <- gray_pixels("Mountain/25.jpg")
mountain26 <- gray_pixels("Mountain/26.jpg")
mountain27 <- gray_pixels("Mountain/27.jpg")
mountain28 <- gray_pixels("Mountain/28.jpg")
mountain29 <- gray_pixels("Mountain/29.jpg")
mountain30 <- gray_pixels("Mountain/30.jpg")
mountain31 <- gray_pixels("Mountain/31.jpg")
mountain32 <- gray_pixels("Mountain/32.jpg")
mountain33 <- gray_pixels("Mountain/33.jpg")
mountain34 <- gray_pixels("Mountain/34.jpg")
mountain35 <- gray_pixels("Mountain/35.jpg")
mountain36 <- gray_pixels("Mountain/36.jpg")
mountain37 <- gray_pixels("Mountain/37.jpg")
mountain38 <- gray_pixels("Mountain/38.jpg")
mountain39 <- gray_pixels("Mountain/39.jpg")
mountain40 <- gray_pixels("Mountain/40.jpg")


type = c(rep("Car",40), rep("Tree",40), rep("Waterfall",40), rep("Beach", 40), rep("Mountain", 40))
images.df <-  data.frame(rbind(car1, car2, car3, car4, car5, car6, car7, car8, car9, car10,car11, car12, car13, car14, car15, car16, car17, car18, car19, car20, car21, car22, car23, car24, car25, car26, car27, car28, car29, car30, car31, car32, car33, car34, car35, car36, car37, car38, car39, car40,
                               tree1, tree2, tree3, tree4, tree5, tree6, tree7, tree8, tree9, tree10,tree11, tree12, tree13, tree14, tree15, tree16, tree17, tree18, tree19, tree20, tree21, tree22, tree23, tree24, tree25, tree26, tree27, tree28, tree29, tree30, tree31, tree32, tree33, tree34, tree35, tree36, tree37, tree38, tree39, tree40,
                               waterfall1, waterfall2, waterfall3, waterfall4, waterfall5, waterfall6, waterfall7, waterfall8, waterfall9, waterfall10,waterfall11, waterfall12, waterfall13, waterfall14, waterfall15, waterfall16, waterfall17, waterfall18, waterfall19, waterfall20, waterfall21, waterfall22, waterfall23, waterfall24, waterfall25, waterfall26, waterfall27, waterfall28, waterfall29, waterfall30, waterfall31, waterfall32, waterfall33, waterfall34, waterfall35, waterfall36, waterfall37, waterfall38, waterfall39, waterfall40,
                               beach1, beach2, beach3, beach4, beach5, beach6, beach7, beach8, beach9, beach10,beach11, beach12, beach13, beach14, beach15, beach16, beach17, beach18, beach19, beach20, beach21, beach22, beach23, beach24, beach25, beach26, beach27, beach28, beach29, beach30, beach31, beach32, beach33, beach34, beach35, beach36, beach37, beach38, beach39, beach40,
                               mountain1, mountain2, mountain3, mountain4, mountain5, mountain6, mountain7, mountain8, mountain9, mountain10, mountain11, mountain12, mountain13, mountain14, mountain15, mountain16, mountain17, mountain18, mountain19, mountain20, mountain21, mountain22, mountain23, mountain24, mountain25, mountain26, mountain27, mountain28, mountain29, mountain30, mountain31, mountain32, mountain33, mountain34, mountain35, mountain36, mountain37, mountain38, mountain39, mountain40),
                              class=type)

images.df_noClass <- images.df[,-ncol(images.df)]
pca.images <- prcomp(images.df_noClass, scale=T)
summary(pca.images)

#Rotation matrix
rots <- pca.images$rotation
dim(rots)
images.mat <- as.matrix(images.df_noClass[,1:ncol(images.df_noClass)])
dim(images.mat)

#Principal Components
pcs <- images.mat %*% rots
dim(pcs)

prcomp.df <- data.frame(pcs[,1:150], class=type)

per <- 0.75 #Training percentage
nr <- nrow(prcomp.df)
train <- sample(1:nr, per * nr, rep=F)
test <- setdiff(1:nr, train)

test.df <- prcomp.df[test,]
train.df <- prcomp.df[train,]

#Random Forest
rf <- randomForest(as.factor(class)~., data=train.df, ntree=500)
pred <- predict(rf, newdata = test.df)
mean(pred != test.df$class)  #Error rate
table(pred, test.df$class)


#SVM with Radial kernel
train <- sample(1:nr, per * nr, rep=F)
test <- setdiff(1:nr, train)
test.df <- prcomp.df[test,]
train.df <- prcomp.df[train,]

tune.svm <-
  tune(svm,as.factor(class) ~.,
       data=train.df,
       kernel="radial",
       scale=F,
       ranges=list(cost=10^seq(-5,5,length=50), gamma=10^seq(-5,5)),
       tunecontrol=tune.control(cross=3))

C <- tune.svm$best.parameters[1]
G <- tune.svm$best.parameters[2]
C
G

svmfit <- svm(as.factor(class) ~., 
              data = train.df,
              kernel = "radial",
              cost = C,
              gamma = G,
              scale = F)
pred.svm <- predict(svmfit, newdata = test.df)
mean(pred.svm != test.df$class)  #Error Rate
table(pred.svm, test.df$class)

####K-Nearest-Neighbour Classification

trainsample <- sample(1:nr, per * nr, rep=F)
testsample <- setdiff(1:nr, train)
prcomp_noClass = prcomp.df[,-ncol(prcomp.df)]
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
knn.df<-as.data.frame(lapply(prcomp_noClass, normalize))
knn_train<-knn.df[trainsample,]
knn_test<-knn.df[testsample,]
knn_train_label<-prcomp.df[train,201]
knn_test_label<-prcomp.df[test,201]
knn_test_pred<-knn(train=knn_train, test=knn_test,cl=knn_train_label,k=10)
mean(knn_test_label != knn_test_pred)
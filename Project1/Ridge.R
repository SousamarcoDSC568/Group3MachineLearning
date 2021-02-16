data <-read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')

names(data) <-c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")

data <- na.omit(data)
data.mat <- model.matrix(MEDV ~ .-1, data=data)
data.mat <- data.mat[, 1:13]
head(data.mat)
set.seed(1)
#install.packages('glmnet')
library(glmnet)

x <- data.mat
y <- data[, 'MEDV']

#install.packages('plotmo')
library(plotmo)
grid <- 10^seq(6, -3, length=10)
ridge.mod <- glmnet(scale(x), y, alpha = 0, lambda = grid, thresh = 1e-2, standardize = TRUE)
plot_glmnet(ridge.mod, xvar = "lambda", label = 13)

cv.out <- cv.glmnet(x, y, alpha=0, nfolds = 10)
cv.out
#Call:  cv.glmnet(x = x, y = y, nfolds = 10, alpha = 0) 
#Measure: Mean-Squared Error 
#Lambda Index Measure    SE Nonzero
#min  0.678   100   24.25 2.546      13
#1se  3.617    82   26.78 3.053      13
plot(cv.out)

best.lambda <- cv.out$lambda.min
best.lambda
#0.6777654

ridge.final <- glmnet(scale(x), y, alpha = 0, lambda = best.lambda, thresh=1e-2, standardsize = TRUE)
predict(ridge.final, type="coefficients", s=best.lambda)
#14 x 1 sparse Matrix of class "dgCMatrix"
#(Intercept) 22.5328063
#CRIM        -0.4433313
#ZN           0.5934338
#INDUS       -0.7647167
#CHAS         0.7494329
#NOX         -1.4139968
#RM           3.2123657
#AGE         -0.3130941
#DIS         -2.5539636
#RAD          0.8424452
#TAX         -0.7510226
#PTRATIO     -1.8099040
#B            0.8974522
#LSTAT       -2.8927874

ridge.pred <- predict(ridge.final, s=best.lambda, newx=x)
print(paste('MSE:', mean((ridge.pred -y)^2)))
#[1] "RMSE: 42046.4971888221"
print(paste('RMSE:', sqrt(mean((ridge.pred - y)^2))))
#[1] "RMSE: 205.052425464373"

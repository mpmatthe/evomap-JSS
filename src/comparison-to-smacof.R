array <- matrix(c(0, 0, 0, 0, 0.07, 0, 0.06, 0.04, 0, 0.07,
                  0, 0, 0.03, 0, 0, 0, 0, 0, 0.08, 0,
                  0, 0.03, 0, 0, 0, 0, 0, 0, 0.01, 0,
                  0, 0, 0, 0, 0, 0.09, 0.03, 0, 0, 0,
                  0.07, 0, 0, 0, 0, 0, 0.01, 0.01, 0, 0,
                  0, 0, 0, 0.09, 0, 0, 0.05, 0, 0, 0,
                  0.06, 0, 0, 0.03, 0.01, 0.05, 0, 0.1, 0, 0.03,
                  0.04, 0, 0, 0, 0.01, 0, 0.1, 0, 0, 0.03,
                  0, 0.08, 0.01, 0, 0, 0, 0, 0, 0, 0,
                  0.07, 0, 0, 0, 0, 0, 0.03, 0.03, 0, 0),
                nrow = 10, ncol = 10, byrow = TRUE)

labels <- c('APPLE INC', 'AT&T INC', 'COMCAST CORP', 'EBAY INC', 'HP INC',
           'INTUIT INC', 'MICROSOFT CORP', 'ORACLE CORP', 'US CELLULAR CORP',
           'WESTERN DIGITAL CORP')

colnames(array) <- labels
rownames(array) <- labels
library(smacof)
D <- smacof::sim2diss(array, method = "reverse")

res <- smacof::smacofSym(D, type = "ordinal")
plot(res, plot.type = "Shepard")

D

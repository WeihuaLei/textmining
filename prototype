#a prototype of text mining
#author -- lei --


#from file.csv
log.csv <- data.frame(lapply(read.csv("Shop_Findings.csv"), as.character), stringsAsFactors=FALSE)
#maintance log text
mainlog <- as.data.frame(log.csv[,7])
#inital reason text
reasonlog <- as.data.frame(log.csv[,c("RFR_DESC","NONCONFORMANCE_TXT")])




#load file
log <- system.file("texts","train",package = "tm")

#some NLP processing(to do)

#create corpus
clog <- Corpus(DataframeSource(mainlog))
##################################
#metadata management
##################################
meta(clog,tag = "serialnum",type = "local") <- log.csv[,1]
meta(clog,tag = "payer",type = "local") <- log.csv[,3]
meta(clog,tag = "isscheduled",type = "local") <- log.csv[,9]
meta(clog,tag = "condition",type = "local") <- log.csv[,11]
meta(clog,tag = "overhaulhours",type = "local") <-log.csv[,12]
meta(clog,tag = "newhours",type = "local") <-log.csv[,13]
#get meta data
meta(clog[as.numeric(clog.lda.vem@documents)],tag="condition")

stopwords <- readLines(con = "stopword.txt")
mydictionary <- c("seal","inspect","turbin","wheel","worn","unit")

extendstopwords <- c("unit","received","complied","have","also","sb","ref","per","sbs","s","b","was","with","honeywell","accomplished","references","start")

#simple custom transformation
#replace some to space
toSpace <-content_transformer(function(x,pattern) gsub(pattern," ",x))
exChange <- content_transformer(function(x,from,to) gsub(from,to,x))

clog<-tm_map(clog,toSpace,"/|:|-|#|@|&|\\(|\\)|\\'")
clog <-tm_map(clog,exChange,"100%","onehundred")
clog <-tm_map(clog,exChange,"bearings","bearing")
clog <-tm_map(clog,exChange,"bulletins","bulletin")
clog <-tm_map(clog,exChange,"contaminated","contamination")
clog <-tm_map(clog,exChange,"failed","failure")
clog <-tm_map(clog,exChange,"gouged","gouge")
clog <-tm_map(clog,exChange,"heavily","heavy")
clog <-tm_map(clog,exChange,"inspected","inspection")
clog <-tm_map(clog,exChange,"leaking","leakage")
clog <-tm_map(clog,exChange,"overhauled","overhaul")
clog <-tm_map(clog,exChange,"repaired","repair")
clog <-tm_map(clog,exChange,"requirements","request")
clog <-tm_map(clog,exChange,"rubbed","rub")
clog <-tm_map(clog,exChange,"seals","seal")
clog <-tm_map(clog,exChange,"shafts","shaft")
clog <-tm_map(clog,exChange,"springs","spring")
clog <-tm_map(clog,exChange,"tested","test")


#some standard preprocessing
#stemming
dictcorpus <- clog
clog <- tm_map(clog,stemDocument)
clog <- tm_map(clog,content_transformer(tolower))
#clog <-tm_map(clog,stemCompletion,dictionary = compdict)
clog <- tm_map(clog,removeWords,stopwords)
clog <- tm_map(clog,removeNumbers)
#remove punctuation
clog <- tm_map(clog,removeWords,extendstopwords)
clog <- tm_map(clog,removePunctuation)
clog <- tm_map(clog,stripWhitespace)

#custom tokenizer
myTokenizer <- function(x)
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
#ngarms tokenizer
NGarmsTokenizer <-  function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
clog2dtm <- DocumentTermMatrix(clog,control = list(tokenize=NGarmsTokenizer,
                                                   #dictionary = dictionary,
                                                bounds=list(local=c(1,Inf)),
                                                wordLengths = c(3,Inf)))
clog2dtm <- clog2dtm[ , which(table(clog2dtm$j) >= 4)]
clog2dtm <- clog2dtm[row_sums(clog2dtm) > 0,]
vocubulary2 <- Terms(clog2dtm)
freq2 <- sort(colSums(as.matrix(clog2dtm)), decreasing=TRUE)
#############################
#filter item through if-idf
##############################
term2_tfidf <- tapply(clog2dtm$v/row_sums(clog2dtm)[clog2dtm$i], clog2dtm$j,mean)*log2(nDocs(clog2dtm)/col_sums(clog2dtm > 0))
summary(term2_tfidf)
clog.idf.dtm2 <- clog2dtm[,term2_tfidf >= 0.2]
clog.idf.dtm2 <- clog.idf.dtm2[row_sums(clog.idf.dtm2) > 0,]
summary(col_sums(clog.idf.dtm2))

library(slam)
#some DTM control
ctrl <-list(tokenize=MC_tokenizer,
            Weighting=weightTf,
            bounds=list(local = c(2, Inf)),
            wordLengths = c(3, 13))
#Document-Term matrix
clogdtm <-DocumentTermMatrix(clog,control = ctrl)
#DTM with if-idf
clogdtmidf <- weightTfIdf(clogdtm)
summary(col_sums(clogdtmidf))
#fliter terms
clogdtm <- clogdtm[ , which(table(clogdtm$j) >= 2)]
clogdtm <- clogdtm[row_sums(clogdtm) > 0,]
vocabulary <- Terms(clogdtm)
summary(col_sums(clogdtm))
#most frequency terms
freq <- sort(colSums(as.matrix(clogdtm)), decreasing=TRUE)
#############################
#filter item through if-idf
##############################

term_tfidf <- tapply(clogdtm$v/row_sums(clogdtm)[clogdtm$i], clogdtm$j,mean)*log2(nDocs(clogdtm)/col_sums(clogdtm > 0))
summary(term_tfidf)
clog.idf.dtm <- clogdtm[,term_tfidf >= 1]
clog.idf.dtm <- clog.idf.dtm[row_sums(clog.idf.dtm) > 0,]
summary(col_sums(clog.idf.dtm))


inspect(clogdtm[1:20,30:40])
highFreq<-findFreqTerms(clogdtm,20,Inf)
findAssocs(clog2dtm,"output shaft",0.1)


#item cloud
set.seed(100)
wordcloud(names(freq),freq,min.freq = 1,max.words = 150,rot.per = 0.1,colors = brewer.pal(6,"Dark2"))




#LDA topic model
#LDA Default control
control_LDA_VEM <- list(estimate.alpha = TRUE, alpha = 50/k,
                        estimate.beta = TRUE,
                        verbose = 0, prefix = tempfile(), save = 0, keep = 0,
                        seed = as.integer(Sys.time()), nstart = 1, best = TRUE,
                        var = list(iter.max = 500, tol = 10^-6),
                        em = list(iter.max = 1000, tol = 10^-4),
                        initialize = "random")
k=30
SEED = 2015
#VEM measurement
clog.lda.vem <-LDA(clogdtm,k = 10,method = "VEM",control = list(seed=SEED))
#Gibbs sampling
clog.lda.gibbs <-LDA(clogdtm,k = 10,method = "Gibbs",control = list(seed = SEED,
                                                                   burnin = 100,
                                                                   thin = 100,
                                                                   iter = 1000))
#CTM with VEM
clog.ctm.vem <-CTM(clogdtm, k = k,
 control = list(seed = SEED, var = list(tol = 10^-4), em = list(tol = 10^-3)))

write.csv(terms(clog.lda.gibbs,122),file="topics_terms_Gibbs.csv")

##########################################
#get the optimum topic numbers#
##########################################
{
library(Rmpfr)
harmonicMean <- function(logLikelihoods, precision=2000L) {
  llMed <- median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}
# generate numerous topic models with different numbers of topics
sequ <- seq(2, 30, 2)
burnin = 1000
iter = 1000
keep = 50

fitted_many <- lapply(sequ, function(k) LDA(clog.idf.dtm2, k = k, method = "Gibbs",control = list(seed = SEED,burnin = burnin, iter = iter, keep = keep)))

# extract logliks from each topic
logLiks_many <- lapply(fitted_many, function(L)  L@logLiks[-c(1:(burnin/keep))])

# compute harmonic means
hm_many <- sapply(logLiks_many, function(h) harmonicMean(h))

# inspect
plot(sequ, hm_many, type = "l")

# compute optimum number of topics
sequ[which.max(hm_many)]
#opt <- fitted_many[which.max(hm_many)][[1]]
}


#########################################################
##############LDA visualization##########################
#########################################################
## Extract the 'guts' of the optimal model
doc.id <- clog.lda.gibbs@wordassignments$i
token.id <- clog.lda.gibbs@wordassignments$j
topic.id <- clog.lda.gibbs@wordassignments$v


phi <- posterior(clog.lda.gibbs)$terms
theta <- clog.lda.gibbs@gamma
vocab <- clog.lda.gibbs@terms
doclength <- (as.numeric(table(doc.id)))
terms.frequency <- as.numeric(table(token.id))


json <- createJSON(phi = phi,
                   theta = theta,
                   doc.length = doclength,
                   vocab = vocab,
                   term.frequency = terms.frequency)

Sys.setenv(GITHUB_PAT = "015557145c5d7a1e58af12ec9d8f0e632c35c328")
require(servr)
require(gistr)
serVis(json,as.gist = T)

################################
####association analysis########
################################



payer <- levels(factor(log.csv[,3]))
frequency <-table(log.csv[,3])


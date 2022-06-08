Sentiment Analysis for Persian Text
================

Sentiment Analysis is a process in which the polarity (positive or
negative) of a given text (comment, tweets, etc.) is determined. In this
project, I have performed sentiment analysis for Persian text by
multiple methodologoes (Naive Bayes, Logistic Regression, Vector Space
Models) in R.

In this project, we are going to develop very basic methodologies for
sentiment analysis including **Logistic Regression**, **Naive Bayes**,
and **Vector Space Classification** and test their performance on a
dataset I have found online.

# loading packages

``` r
library(rwhatsapp)
library(PersianStemmer)
```

# Loading & Preparing the data

loading the DigiKala data related to iPhone that I have found online.

``` r
dat = read.csv("Data.txt", header = T, encoding = "UTF-8")
#dat=read_excel("Data.xlsx")
dat = na.omit(dat)
```

By considering the label “1” as “Positive” and the rest of the labels as
“Negative”, we are preparing the data for creating a model for detecting
the sentiments from a comment.

``` r
Corpus = lookup_emoji(dat$Text, text_field = "text")
Text = Corpus$text
Emoji = Corpus$emoji
Y = as.numeric(dat$Suggestion==1)
```

Defining functions for text cleaning, building up a dictionary, and
extracitng features. First, the function that prepares the text for the
analysis.

``` r
RefineText<- function(tex){
  n = length(tex)
  TextAd = c()
  
  #progress bar
  pb = txtProgressBar(min = 0, max = n, initial = 0,
                      style = 3, width = 50, char = "=") 
  
  for (i in 1:n) {
    tex[i] = RefineChars(tex[i]) 
    tex[i] = gsub("[\r\n]", "", tex[i])
    tex[i] = gsub('[[:punct:] ]+',' ', tex[i])
    tex[i] = RefineChars(tex[i])
        
    #,"@","$","%","&","*","،",".....","..."
    if (!(is.na(tex[i]))){
      if (all(tex[i]!=c(""," ","،","،,","‌"))){
        TextAd[i] = PerStem(tex[i], NoEnglish = F, NoNumbers = F,
                        NoStopwords = T, NoPunctuation = T,
                        StemVerbs = T, NoPreSuffix = T,
                    Context = T, StemBrokenPlurals = T,
                    Transliteration = F)
      }
    }
    setTxtProgressBar(pb,i)
  }
return(TextAd)
}
```

Now, it is time to build up a dictionary.

``` r
BuildFreqs <- function(corpus,y) {
  n = length(y)
  d = 1 # dictionary word numerator
  freqs=matrix(c(0,0),ncol = 2) #frequencies for each word
  colnames(freqs)=c("Neg","Pos")
  Dict=c() 
  t=c()
  e=c()
  tex = corpus$text
  emoji = corpus$emoji
  #refining Text removing numbers, stop words, punctuation, and so
  #Progress bar
  pb = txtProgressBar(min = 0, max = n, initial = 0, style = 3, width = 50, char = "=")
  for (i in 1:n) {
    t = sort(table(unlist(strsplit(tex[i], " "))), decreasing = TRUE)
  
    #Building freqs for words
    for (j in 1:length(t)) {
      cond = (Dict == names(t)[j])
      if  (any(cond)) {
        wh = which(cond)
        freqs[wh+1,(y[i]+1)] = freqs[wh+1,(y[i]+1)]+t(j) # Adding Frequency
      } else if(is.null(names(t))==F) {
        Dict[d] = names(t)[j]
        if (y[i]==1) {
          freqs = rbind(freqs,c(0,1)) # Defining Frequency
        } else {
          freqs = rbind(freqs,c(1,0)) # Defining Frequency 
        }
        d = d+1
      }
    }
  
    if(is.null(emoji[[i]])==F){
      e = table(emoji[i])
    
      #Building frequencies for emojis
      for (j in 1:length(e)) {
        cond = (Dict == names(e)[j])
        if  (any(cond)) {
          wh = which(cond)
          freqs[wh+1,(y[i]+1)] = freqs[wh+1,(y[i]+1)]+e[j] 
          # Adding Frequency
        } else if(is.null(names(e))==F) {
            Dict[d] = names(e)[j]
            if (y[i]==1) {
              freqs = rbind(freqs,c(0,1)) # Defining Frequency
            } else {
              freqs = rbind(freqs,c(1,0)) # Defining Frequency 
            }
            d = d+1
        }
      }
    }
    setTxtProgressBar(pb,i)
  }
Dictionary = list("Words"=Dict,"Frequencies"=freqs[-1,])
  return(Dictionary)
}
```

Finally, a function that uses the dictionary to extract features.

``` r
ExtractFeatures <- function(corpus, dict){
  
  tex = corpus$text
  emoji = corpus$emoji
  
  Words = dict$Words
  dict$Frequencies = dict$Frequencies/rowSums(dict$Frequencies)
  Frequencies = dict$Frequencies
  
  #Progress bar
  n = length(tex)
  pb = txtProgressBar(min = 0, max = n, initial = 0, style = 3,
width = 50, char = "=")
  
   x = matrix(rep(0,n*3),nrow =n ,ncol=3)
  for (i in 1:n) {
    wordlist = names(sort(table(unlist(strsplit(tex[i], " "))), 
                          decreasing = TRUE))
    if(is.null(emoji[[i]])==F){append(wordlist,table(emoji[i]))}
    x[i,1] = 1 #bias term is set to 1
    # loop through each word in the list of words
    for (word in wordlist){ 
      if (any(Words==word)) {
        # increment the word count for the neutral label 0
        x[i,2] = x[i,2]+Frequencies[which(Words==word),1]
       # increment the word count for the positive label 1
        x[i,3] = x[i,3]+Frequencies[which(Words==word),2]
      }
    }
    setTxtProgressBar(pb,i)
  }
 return(x) 
}
```

Using the functions defined, we are going to clean our Persian text,
then build a dictionary, and extract features from the text.

``` r
TextAd = RefineText(Text)
Dictionary = BuildFreqs(list("text"=TextAd,"emoji"=Emoji),Y)
X = ExtractFeatures(list("text"=TextAd,"emoji"=Emoji), Dictionary)
```

# Checking the features interpretability

To see, weather it is possible to to classify our text using the
features we built, We must check the below scatter plot.

``` r
mylabel <- c("Train Pos", "Train Neg")
colors <- c("blue", "red")
xlabel = "Sum of Negative Words"
ylabel = "Sum of Positive Words"
plot(x = X[Y==0,2], y = X[Y==0,3], xlab = xlabel, ylab = ylabel,col="red")
points(x = X[Y==1,2], y = X[Y==1,3], col="green")
```

![](Sentiment-Analysis-for-Persian-Text-in-R/figures/000012.png)<!-- -->

The first classification method that is widely being used as a trivial
and easy-to-implement methodology is Naive Bayes. In the following lines
of code, we are going to build two functions by which we can use this
method for our data.

``` r
ExtractSense=function(corpus, dict){
  
  tex = corpus$text
  emoji = corpus$emoji
  
  Words = dict$Words
  Frequencies = dict$Frequencies
  
  d = length(dict$Words)
  n = length(tex)
  
  #Laplacian Smoothing
  Freqs = 1/(colSums(Frequencies)+d)*(t(Frequencies)+1)
  Freqs = t(Freqs)
  
  p_w_pos = Freqs[,2]
  p_w_neg = Freqs[,1]
  
  loglikelihood = log(p_w_pos/p_w_neg)
  
  p=rep(0,n)
  
  pb = txtProgressBar(min = 0, max = n, initial = 0, style = 3, width = 50,
                      char = "=")
  
  for(i in 1:n){
    
    wordlist = sort(table(unlist(strsplit(tex[i], " "))), decreasing = TRUE)
    if(is.null(emoji[[i]])==F){append(wordlist,table(emoji[i]))}
    
    for (j in 1:length(wordlist)){
      if  (any(Words == names(wordlist)[j])) {
        p[i]= p[i]+wordlist[j]*loglikelihood[Words == names(wordlist)[j]]
      }
    }
    setTxtProgressBar(pb,i)
  }
  return(p)
}
```

And, the predictior function will be defined like this:

``` r
NaiveBayesPredictor<-function(p,y){
  
  D_pos = sum(y==1)
  D_neg = sum(y==0)
  
  logprior = log(D_pos/D_neg)
  
  p = p+rep(logprior,length(p))
  
  if (length(p)==length(y)){
    yhat = (p>0)
    accuracy = mean(y==yhat)
  }
  
  return(list("Yhat"=yhat,"Accuracy"=accuracy))
}
```

Now, we want to use this method on our data to build a feature called
“Sense” that measures the intensity of each comment alongside their
polarity.

``` r
Sense = ExtractSense(list("text"=TextAd,"emoji"=Emoji), Dictionary)
```

We can see below how this feature looks like.

``` r
pos = which(Sense>1)
neg = which(Sense<1)

plot(x = 1:length(Sense), y = Sense, type = 'bar')
points(x = 1:length(Sense), y= Sense*(Sense>1), col="green")
points(x = 1:length(Sense), y= Sense*(Sense<1), col="red")
```

![](Sentiment-Analysis-for-Persian-Text-in-R/figures/000016.png)<!-- -->

# Checking the Accuracies for each Methodology

The accuracy of Naive Bayes classifier on the train dataset can be
calculated as follow.

``` r
NB = NaiveBayesPredictor(Sense,Y)
yhat = NB$Yhat
NBAccuracy=c()
NBAccuracy["Positive"] = 
  mean(na.omit(Y[Y==1]==yhat[Y==1]))
NBAccuracy["Negative"] = 
  mean(na.omit(Y[Y==0]==yhat[Y==0]))
cat(paste0("Overal Accuracy: \n"),mean(yhat==Y),
    paste0("\nAccuracy for different sentiments: \n"), names(NBAccuracy),
    paste0(" \n"),NBAccuracy)
```

    ## Overal Accuracy: 
    ##  0.8730451 
    ## Accuracy for different sentiments: 
    ##  Positive Negative  
    ##  0.9286314 0.7224118

The second classifier is based on vector space models. We are going to
build such a classifier in the following lines of code.

``` r
VectorSpaceModel= function(x,y){
  
  pos_center = colMeans(x[y==1,])
  neg_center = colMeans(x[y==0,])
  centers = cbind(pos_center[-1],neg_center[-1])
  dot_prods = x[,-1]%*%centers
  normX = apply(x[,-1],1,function(x)(sqrt(sum(x^2))))
  normCenter = apply(centers,2,function(x)(sqrt(sum(x^2))))
  
  angel_cosine = as.matrix(1/normX)%*%t(as.matrix(1/normCenter))*(dot_prods)
  
  yhat = apply(angel_cosine, 1, function(x) which(max(x)==x))
  yhat[yhat==2] = 0
  return(list("Yhat" = yhat, "Centers" = centers))
}
```

The next function is going to calculates (predicts) the polarity for
each comment. Plus, if we input the dependent variable (y) as well it
gives us the accuracy of the predictions.

``` r
VectorSpacePredictor = function(x, y =NA, centers){
  dot_prods = x[,-1]%*%centers
  normX = apply(x[,-1], 1, function(x)(sqrt(sum(x^2))))
  normCenter = apply(centers, 2, function(x)(sqrt(sum(x^2))))
  
  angel_cosine = as.matrix(1/normX)%*%t(as.matrix(1/normCenter))*(dot_prods)
  
  yhat = apply(angel_cosine, 1, function(x) which(max(x)==x))
  yhat[yhat==2] = 0
  
  accuracy = NA
  
  if(is.na(y)==FALSE && length(y)==length(yhat)){
    accuracy = mean(y==yhat)
  }
  return(list("Yhat"=yhat,"Accuracy"=accuracy))
}
```

Train accuracy for Vector Space classifier is as follows.

``` r
fitVS = VectorSpaceModel(X,Y)
yhat = fitVS$Yhat
VSAccuracy = c()
VSAccuracy["Positive"] = 
  mean(na.omit(Y[Y==1]==yhat[Y==1]))
VSAccuracy["Negative"] = 
  mean(na.omit(Y[Y==0]==yhat[Y==0]))
cat(paste0("Overal Accuracy: \n"),mean(yhat==Y),
    paste0("\nAccuracy for different sentiments: \n"), names(VSAccuracy),
    paste0(" \n"),VSAccuracy)
```

    ## Overal Accuracy: 
    ##  0.9024839 
    ## Accuracy for different sentiments: 
    ##  Positive Negative  
    ##  0.9601175 0.7463026

The logistic regression classifier accuracy on this dataset can be
calculated as follow.

``` r
fitGLM = glm(Y~X[,-1],family="binomial")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
yhat = round(predict(fitGLM,as.data.frame(X),type="response"),0)
GLMAccuracy = c()
GLMAccuracy["Positive"] = 
  mean(na.omit(Y[Y==1]==yhat[Y==1]))
GLMAccuracy["Negative"] = 
  mean(na.omit(Y[Y==0]==yhat[Y==0]))
cat(paste0("Overal Accuracy: \n"),mean(yhat==Y),
    paste0("\nAccuracy for different sentiments: \n"), names(GLMAccuracy),
    paste0(" \n"),GLMAccuracy)
```

    ## Overal Accuracy: 
    ##  0.9101503 
    ## Accuracy for different sentiments: 
    ##  Positive Negative  
    ##  0.972712 0.7406143

# Comparison of the three methodologies in a simulation

It is not obvious which method performs better on this dataset. So, we
run a simulation of 1000 iterations each time we are going to randomly
split the dataset into train and test. Then we build our models based on
the train sets and test their performances on the test sets. the results
we be stored in a matrix and after the simulation will be visualized.

So, the simulation for the defined methodologies is going to be like
this.

``` r
PosX = X[Y==1,]
NegX = X[Y==0,]
PosSense = Sense[Y==1]
NegSense = Sense[Y==0]
nP = dim(PosX)[1]
nN = dim(NegX)[1]
Accuracies = 0
Error = matrix(rep(0,3*1000), ncol=3)
colnames(Error) = c("Vector Space Classifier Error",
                  "Naive Bayes Classifier Error",
                  "Logistic Regression Classifier Error")
for(iter in 1:1000){
  indexP = sample(nP, ceiling(nP*0.9), replace = F)
  indexN = sample(nN, ceiling(nN*0.9), replace = F)
  TrainXPos = PosX[indexP,]
  TrainXNeg = NegX[indexN,]
  TestXPos = PosX[-indexP,]
  TestXNeg = NegX[-indexN,]
  
  TestSensePos = as.matrix(PosSense[-indexP])
  TestSenseNeg = as.matrix(NegSense[-indexN])
  
  SenseTest = append(TestSensePos, TestSenseNeg)
  
  XTrain = rbind(TrainXPos, TrainXNeg)
  XTest = rbind(TestXPos, TestXNeg)
  TrainY = c(rep(1, dim(TrainXPos)[1]), rep(0, dim(TrainXNeg)[1]))
  TestY = c(rep(1, dim(TestXPos)[1]), rep(0, dim(TestXNeg)[1]))
  fitVS = VectorSpaceModel(XTrain, TrainY)
  VS = VectorSpacePredictor(XTest, TestY, fitVS$Centers)
  
  NB = NaiveBayesPredictor(SenseTest, TestY)
  
  fitGLM = glm(TrainY~XTrain[,-1], family = "binomial")
  labelGLM = predict(fitGLM, as.data.frame(XTest), type="response")
  
  Error[iter,] = c(1-VS$Accuracy, 1-NB$Accuracy, 1-(mean(round(labelGLM, 0)==TestY)))
}
```

So, in the following lines of code, we are going to build a box plot to
compare the error of each method.

``` r
boxplot(Error)
```

![](Sentiment-Analysis-for-Persian-Text-in-R/figures/000033.png)<!-- -->

It seems obvious now that vector space classifier performs slightly
better than naive Bayes and significantly better than Logistic
regression classifier.

Sentiment Analysis for Persian Text
================

Sentiment Analysis is a process in which the polarity (positive or
negative) of a given text (comment, tweets, etc.) is determined. In what follows, I will explain how you can perform sentiment analysis for Persian text based on multiple methodologies ( **Naive Bayes**, **Logistic Regression**, **Vector Space Models** ) in [R](https://www.r-project.org/). Furthermore, I will evaluate their performance on a dataset I have found online. Let us begin!

# loading packages
First off, there are multiple packages that I have used for the entire coding section. make sure that you have them already installed.
``` r
install.packages("rwhatsapp")
install.packages("PersianStemmer")
```
Then, it is time to load the packages.
``` r
library(rwhatsapp)
library(PersianStemmer)
```

# Loading & Preparing the data

At this stage, I load the DigiKala data related to the iPhone that I have found online.

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

In Sentiment Analysis, there is a text cleaning section (including removing handles and URLs, removing punctuation, tokenizing, etc.) that the package **PersianStemmer** can handle for us. However, the package **PersianStemmer** cannot get a vector of strings and return it in a format of a cleaned vector. Therefore, I am going to build a function based on the package so it can handle this issue. This function prepares the text for the analysis.

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

Now, it is time to build a list of all the words that have appeared throughout the corpus along with two measurements. The number of times a particular word appeared in positive and negative comments. This list is called a dictionary. In the next step, we are going to create one.

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

# Naive Bayes Classifier

The first classification method that is widely being used as a trivial and easy-to-implement methodology is Naive Bayes. In the following lines of code, we are going to build two functions by which we can use this method for our data. First function is a function that uses the corpus and the dictionary that we just built. This function returns a number that represents the overall sentiment or polarity (p) expressed within a comment.

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

The second one is the predictor function is going to return the accuracy and the predicted polarity for each text.

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

Now, we want to use these functions on our data to build a feature called
“Sense” that measures the intensity of each comment alongside their
polarity.

``` r
TextAd = RefineText(Text)
Dictionary = BuildFreqs(list("text"=TextAd,"emoji"=Emoji),Y)
Sense = ExtractSense(list("text"=TextAd,"emoji"=Emoji), Dictionary)
```

Now, we can see below how these features are looking like.

``` r
pos = which(Sense>1)
neg = which(Sense<1)

plot(x = 1:length(Sense), y = Sense, type = 'bar')
points(x = 1:length(Sense), y= Sense*(Y==1), col="green")
points(x = 1:length(Sense), y= Sense*(Y==0), col="red")
```

![NaiveBayes](https://github.com/alcstat/Sentiment-Analysis-for-Persian-Text-in-R/blob/main/figures/NaiveBayes.png)<!-- -->
(Overall Sentiments for each comment based on Naive Bayes approach)

Naive Bayes seems to perform well enough for the discrimination task. The accuracy of the Naive Bayes classifier on the training dataset can be
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

# Vector Space Classifier

The second classifier is based on Vector Space models. We are going to build such a classifier in the following lines of code. Based on the Vector Space approach and the dictionary that we have just built, we are going to turn each comment within the whole body of text which is called corpus into two quantities. The first is the summation of positive weights of all the words appeared within each comment, and the second one is the summation of negative weights.

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

Using the functions defined, we are going to clean our Persian text, then build a dictionary, and extract features from the text.

``` r
X = ExtractFeatures(list("text"=TextAd,"emoji"=Emoji), Dictionary)
```

To see whether it is possible to classify our text using the features we built, I am going to visualize each comment based on the calculated features with green color for positive comments and red color for negative comments.

``` r
mylabel = c("Train Pos", "Train Neg")
colors = c("blue", "red")
xlabel = "Sum of Negative Words"
ylabel = "Sum of Positive Words"
plot(x = X[Y==0,2], y = X[Y==0,3], xlab = xlabel, ylab = ylabel,col="red")
points(x = X[Y==1,2], y = X[Y==1,3], col="green")
```

![VectorSpace](https://github.com/alcstat/Sentiment-Analysis-for-Persian-Text-in-R/blob/main/figures/VectorSpace.png)<!-- -->
(Polarity defined by Vector Space approach for all comments within the corpus)

It can be seen that based on the two features positive and negative comments can be discreminated.


In the following, I am going to build a function based on the Vector Space approach. In this function first, I am going to classify a new text as positive, if it is closer to center of positive comments by angel.

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

The next function is going to calculate (predict) the polarity for each comment. Plus, if we input the dependent variable (y), it gives us the accuracy of the predictions as well.

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

Train accuracy for the Vector Space classifier is as follows.

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

# Logistic regression Classifier

The logistic regression classifier accuracy for this dataset can be calculated as follow.

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

It is not obvious which method performs better on this dataset. So, we run a simulation of 1000 iterations each time we are going to randomly split the dataset into train and test. Then we build our models based on the train sets and test their performances on the test sets. the results are stored in a matrix and after the simulation will be visualized.

So, the simulation for the defined methodologies is going to be like this.

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

So, in the following lines of code, we are going to build a box plot to compare the error of each method.

``` r
boxplot(Error)
```

![Comparison](https://github.com/alcstat/Sentiment-Analysis-for-Persian-Text-in-R/blob/main/figures/Comparison.png)<!-- -->
(Comparison of the three methodologies by their performances)

It seems obvious now that the Vector Space classifier performs slightly
better than Naive Bayes. Also, the Logistic regression classifier has a poor performance.

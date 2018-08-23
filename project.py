# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:45:42 2018

@author: satya naidu
"""
# import all required libraries
import numpy as np
import pandas as pd

# import libraries for feature engineering and model training
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# import libraries for shallow neural network
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


# import libraries for results visualization

import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
# dataset preperation, text data import using pandas 
list_=[]

for i in range(1,1001):
    if i <10:
        with open("Text000"+str(i)+".txt", 'rb') as myfile:
            data=myfile.read()
            list_.append(data)
    elif (i>=10) and (i <100):
        with open("Text00"+str(i)+".txt", 'rb') as myfile:
            data=myfile.read()
            list_.append(data)
    elif (i>=100)and (i <1000):
        with open("Text0"+str(i)+".txt", 'rb') as myfile:
            data=myfile.read()
            list_.append(data)
    elif (i==1000):
        with open("Text"+str(i)+".txt", 'rb') as myfile:
            data=myfile.read()
            list_.append(data)
            
df= pd.DataFrame({'Content':list_})
df["Content"]=df["Content"].str.decode("utf-8", errors='replace')



# cleaning data from dataframe 
# import libraries to clean data
import re
from nltk.corpus import stopwords

# stemming and lemmatization are the basic text processing methods for english text
from nltk.stem import WordNetLemmatizer

# cleaning data using lemmatization
corpus=[]
for i in range(0,1000):
    article = re.sub('[^a-zA-Z]',' ',df['Content'][i])
    article= article.lower()
    article=article.split()
    lemmatizer= WordNetLemmatizer()
    article=[lemmatizer.lemmatize(word) for word in article if not word in set(stopwords.words('english'))]
    article= ' '.join(article)
    corpus.append(article)

clean_df = pd.DataFrame({'Content':corpus})
clean_df["Label"]=""
clean_df.loc[0:636,"Label"]="objective"
clean_df.loc[637:,"Label"]="subjective"


X= clean_df.iloc[:,0].values
y= clean_df.iloc[:,1].values

# split data 

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y)
from sklearn.preprocessing import LabelEncoder
labelencoder_y= LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)
y_test= labelencoder_y.fit_transform(y_test)
    
# feature engineering 

#### Creating the Bag of Words model--- using count vectorizer ######
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(clean_df["Content"])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(X_train)
xvalid_count =  count_vect.transform(X_test)


##### using TF-IDF vectors ##### 

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(clean_df['Content'])
xtrain_tfidf =  tfidf_vect.transform(X_train)
xtest_tfidf =  tfidf_vect.transform(X_test)


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(clean_df['Content'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)


###### Topic models as features #####

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))



#### Model Building  #### 

def train_model(classifier, feature_vector_train,label,feature_vector_valid,is_neural_net=False):
    classifier.fit(feature_vector_train,label)
    predictions= classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions,y_test)

#shallow neural network
def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

# only for count vectors
mean=[]
std=[]
lists=['NB_Count_Vectors','LR_Count_Vectors','SVM_Count_Vectors']
estimators=[naive_bayes.MultinomialNB(),linear_model.LogisticRegression(),svm.SVC()]
for model in estimators:
    accuracy=train_model(model, xtrain_count, y_train, xvalid_count)
    mean.append(accuracy)
model_values1 = pd.DataFrame({"Model":lists,"accuracy":mean},index=None)


classifier = create_model_architecture(xtrain_count.shape[1])
acc = train_model(classifier, xtrain_count, y_train, xvalid_count, is_neural_net=True)
model_values1.loc[len(model_values1)]=["ANN_Count_vectors",acc]

# only for TFIDF vectors on words
mean=[]
std=[]
lists=['NB_TFIDF_Word_Vectors','LR_TFIDF_Word_Vectors','SVM_TFIDF_Word_Vectors']
estimators=[naive_bayes.MultinomialNB(),linear_model.LogisticRegression(),svm.SVC()]
for model in estimators:
    accuracy=train_model(model, xtrain_tfidf, y_train, xtest_tfidf)
    mean.append(accuracy)
model_values2 = pd.DataFrame({"Model":lists,"accuracy":mean})

classifier = create_model_architecture(xtrain_tfidf.shape[1])
acc = train_model(classifier, xtrain_tfidf, y_train, xtest_tfidf, is_neural_net=True)
model_values2.loc[len(model_values2)]=["ANN_TFIDF_Word_vectors",acc]

# only for TFIDF vectors on NGrams
mean=[]
std=[]
lists=['NB_TFIDF_NGram_Vectors','LR_TFIDF_NGram_Vectors','SVM_TFIDF_NGram_Vectors']
estimators=[naive_bayes.MultinomialNB(),linear_model.LogisticRegression(),svm.SVC()]
for model in estimators:
    accuracy=train_model(model, xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)
    mean.append(accuracy)
model_values3 = pd.DataFrame({"Model":lists,"accuracy":mean})

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
acc = train_model(classifier, xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram, is_neural_net=True)
model_values3.loc[len(model_values3)]=["ANN_TFIDF_NGram_vectors",acc]

print (tabulate(model_values1, floatfmt=".4f", headers=("model", 'score'),showindex=None))
print (tabulate(model_values2, floatfmt=".4f", headers=("model", 'score'),showindex=None))
print (tabulate(model_values3, floatfmt=".4f", headers=("model", 'score'),showindex=None))

# visualizing results
sns.set_style("darkgrid")

plt.figure(figsize=(15, 6))
sns.barplot(x="Model",y="accuracy",data=model_values1)
sns.barplot(x="Model",y="accuracy",data=model_values2)
sns.barplot(x="Model",y="accuracy",data=model_values3)


plt.figure(figsize=(12, 6))
fig = sns.pointplot(x='Model', y='accuracy', data=model_values1)
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="Models trained")
fig.set(title="accuracy on Count Vectors")



plt.figure(figsize=(12, 6))
fig = sns.pointplot(x='Model', y='accuracy', data=model_values2)
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="Models trained")
fig.set(title="accuracy on Count Vectors")


plt.figure(figsize=(12, 6))
fig = sns.pointplot(x='Model', y='accuracy', data=model_values3)
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="Models trained")
fig.set(title="accuracy on Count Vectors")
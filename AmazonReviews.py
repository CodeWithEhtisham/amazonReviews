#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


# In[2]:


# read the data
df=pd.read_csv('processed_reviews_split_surnamesA_minimal.csv')


# In[3]:


# print top 5 row
df.head()


# In[4]:


# drop extra columns
df.drop(df.columns[5:],inplace=True,axis=1)


# In[5]:


# print the shape of the data
df.shape


# In[6]:


# check Nan values in the data
df.isnull().sum()


# In[7]:


# drop nan values
df.dropna(inplace=True)


# In[8]:


# check Nan values in the data
# new all empty values removed
df.isnull().sum()


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


# information about the columns
df.info()


# In[13]:


# check total rating count
df['review_score'].value_counts()


# In[14]:


# remove wrong rating
df2 = df[df['review_score'].isin(['0','1','2','3','4','5'])]


# In[15]:


# mask = np.logical_or.reduce([(df['review_score'] == cond) for cond in [-1,0,1,2,3,4,5]])
# df2 = df[mask]


# In[16]:


df2


# In[17]:


# check total rating per class
# distibutions
df2.review_score.value_counts()


# In[18]:


sns.countplot(data=df2,x='review_score')


# In[19]:


df2.describe()


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer


# In[21]:


#create a stemmer
stemmer = SnowballStemmer("english")


# In[22]:


#define our own tokenizing function that we will pass into the TFIDFVectorizer.
def tokens(x):
    x = x.split()
    stems = []
    [stems.append(stemmer.stem(word)) for word in x]
    return stems


# In[23]:


#define the vectorizer
vectorizer = TfidfVectorizer(tokenizer = tokens, stop_words = 'english', ngram_range=(1, 1), min_df = 0.01)
#fit the vectorizers to the data.
# saperate the features from the dataset
features = vectorizer.fit_transform(df2['text'])


# In[24]:


features[0][0]


# In[ ]:





# In[25]:


# saperate the target column from teh dataset
target=df2.review_score.values


# In[26]:


target


# In[27]:


target.astype(int)


# In[28]:


from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2, random_state=42)


# # First Model for rating classification (RandomForestClassifier)

# In[29]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#make the grid search object
gs2 = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={ 
    'n_estimators': [200]
},
    cv=StratifiedKFold(n_splits=5),
    
)

#fit the grid search object to our new dataset
print ('Fitting grid search...')
gs2.fit(X_train, y_train)
print ("Grid search fitted.")


# In[30]:


# Make predictions for the test set
y_pred_test = gs2.predict(X_test)


# In[31]:


# View accuracy score
accuracy_score(y_test, y_pred_test)


# In[32]:


# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)


# In[33]:


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['0','1','2','3','4','5']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[34]:


# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))


# # Second Model for rating classification (LogisticRegression)

# In[35]:


from sklearn.linear_model import LogisticRegression
gs2 = GridSearchCV(
    estimator=LogisticRegression(solver='lbfgs', max_iter=1000),
    cv=StratifiedKFold(n_splits=5),
    param_grid={'class_weight': [None, 'balanced']},
)

#fit the grid search object to our new dataset
print ('Fitting grid search...')
gs2.fit(X_train, y_train)
print ("Grid search fitted.")


# In[36]:


# Make predictions for the test set
y_pred_test = gs2.predict(X_test)


# In[37]:


# View accuracy score
accuracy_score(y_test, y_pred_test)


# In[38]:


# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)


# In[39]:


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['0','1','2','3','4','5']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# # Whether a product is from the category “Video Games” (“video_games”) or “Musical Instrument” (“musical_instrument”).

# In[40]:


df2.head()


# In[41]:


sns.countplot(data=df2,x='product_category')


# In[42]:


df2.isnull().sum()


# In[43]:


# change Categorical to numerical data
from sklearn.preprocessing import LabelEncoder


# In[44]:


lebel=LabelEncoder()


# In[45]:


# convert musical_instruments to 0
# convert video_games to 1
df2['product_category']=lebel.fit_transform(df2.product_category.values)


# In[46]:


df2.head()


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(features,df2.product_category.values, test_size=0.2, random_state=42)


# #  First Model for Category classification (RandomForestClassifier)

# In[48]:


#make the grid search object
gs2 = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={ 
    'n_estimators': [200]
},
    cv=StratifiedKFold(n_splits=5),
    
)

#fit the grid search object to our new dataset
print ('Fitting grid search...')
gs2.fit(X_train, y_train)
print ("Grid search fitted.")


# In[49]:


# Make predictions for the test set
y_pred_test = gs2.predict(X_test)


# In[50]:


# View accuracy score
accuracy_score(y_test, y_pred_test)


# In[51]:


# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)


# In[52]:


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['0','1']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# # Second Model for Category classification (LogisticRegression)

# In[53]:


from sklearn.linear_model import LogisticRegression
gs2 = GridSearchCV(
    estimator=LogisticRegression(solver='lbfgs', max_iter=1000),
    cv=StratifiedKFold(n_splits=5),
    param_grid={'class_weight': [None, 'balanced']},
)

#fit the grid search object to our new dataset
print ('Fitting grid search...')
gs2.fit(X_train, y_train)
print ("Grid search fitted.")


# In[54]:


# Make predictions for the test set
y_pred_test = gs2.predict(X_test)


# In[55]:


# View accuracy score
accuracy_score(y_test, y_pred_test)


# In[56]:


# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)


# In[57]:


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['0','1']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[ ]:





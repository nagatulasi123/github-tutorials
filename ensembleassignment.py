#!/usr/bin/env python
# coding: utf-8

# # Problem Statements:¶
# * Given is the diabetes dataset. Build an ensemble model to correctly classify the outcome variable and improve your model prediction by using GridSearchCV. You must apply Bagging, Boosting, Stacking, and Voting on the dataset.

# In[1]:


# import the required libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# In[2]:


# import the dataset 
diabeties =pd.read_csv(r"C:\Users\chand\Desktop\ML ASSIGNMENTS\Diabetes.csv")
diabeties 


# In[3]:


# Rename the columns


# In[4]:


diabeties.rename(columns={' Number of times pregnant':'pregnent', ' Plasma glucose concentration':'glucose',
       ' Diastolic blood pressure':'bloodpressure', ' Triceps skin fold thickness':'skinthickness',
       ' 2-Hour serum insulin':'insulin', ' Body mass index':'bmi',
       ' Diabetes pedigree function':'diabetespedigreefunction', ' Age (years)':'age', ' Class variable':'classvariable'},inplace=True)


# In[5]:


diabeties['outcome']=diabeties['classvariable'].replace({'YES': '1', 'NO': '0'})


# In[6]:


diabeties


# In[7]:


diabeties.dtypes


# In[8]:


diabeties.isnull().sum()


# # Exploratory Data Analysis

# In[9]:


# Counting the number of people with and without diabetes

ax = sns.countplot(diabeties['outcome'])

ax.yaxis.grid()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)

plt.show()


# # The number of people with diabetes is about half of the number of those without.

# In[10]:


# Plotting the distribution of the variables

diabeties1 = diabeties.drop('outcome', axis=1, inplace=False)

ax = diabeties1.hist(figsize = (10,10))
plt.show()


# In[18]:


# Pair Plot allows us to plot the KDE - such as the one above - for all the combination of variables
# It also yields scatter plots

ax = sns.pairplot(diabeties, hue = 'outcome')


# * We can conclude that older age, high glucose concentration, and high number pregnancies are associated with diabetes.
# * However, the association doesn't seem to be strong.

# In[19]:


# Using correlation matrix to understand the correlation between variables
# We can exclude the vraibales, from the model, that have strong correlation with the other variables

df_corr = diabeties1.corr()

sns.set(style="white")
mask = np.zeros_like(df_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 6))

cmap = sns.diverging_palette(255, 133, as_cmap=True)

sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=0.5, annot=True)

plt.yticks(rotation=0, ha="right")
plt.xticks(rotation=90, ha="center")

plt.show()


# # As none of the variables has a strong correlation with any other, we can't eliminate any of the variables

# In[21]:


diabeties


# In[23]:


diabeties.drop(['classvariable'],axis=1,inplace=True)


# In[24]:


# We will scale the features using standardization

y = diabeties['outcome']
X = diabeties.drop('outcome', axis=1, inplace=False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['pregnancies','glucose','bloodPressure','skinthickness',
                             'insulin','bmi','diabetespedigreefunction','age'])
X.head()


# In[25]:


y


# In[26]:


# Models
# A
# 1. Logistic Regression
# 2. Decision Tree
# 3. KNN
# 4. Random Forest
# 5. AdaBoosting

# B
# 1. Voting Classifier


# In[27]:


# We will split the data set into train and test (30% of the original dataframe)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[28]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[30]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred_logreg = logreg.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_logreg))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_logreg))


# In[31]:


# Decision Tree

from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier()
dectree.fit(X_train,y_train)


# In[32]:


y_pred_dectree = dectree.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_dectree))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_dectree))


# In[33]:


# KNN
# We will check the accuracy with different number of neighbors

from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(1,15)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
    
plt.title('k-NN Accuracy for different number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[34]:


# The accuracy on the test dataset is maximum with 13 neighbors

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)


# In[35]:


y_pred_knn = knn.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_knn))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_knn))


# In[36]:


from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators=1000, random_state=0)
ranfor.fit(X_train, y_train)


# In[37]:


y_pred_ranfor = ranfor.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_ranfor))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_ranfor))


# In[38]:


from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(n_estimators=1000)
abc.fit(X_train, y_train)


# In[39]:


y_pred_abc = abc.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_abc))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_abc))


# *  We have already used two ensemble methods - Random Forests (Averaging) and Adaptive Boosting (Boosting) 
# * To improve accuracy, we will combine different classifiers using Voting Classifier, which is also an ensemble method. 

# In[40]:


# Voting Classifier without weights

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('logreg',logreg),('dectree',dectree),('ranfor',ranfor),('knn',knn),('abc',abc)], 
                      voting='soft')
vc.fit(X_train, y_train)


# In[41]:


y_pred_vc = vc.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc))


# * The accuracy of Voting Calssifier is more than any of the other individual classsifiers

# * Now, we will use Voting classifier with weights
# * We will assign more weight to the classifiers with better accuracy

# In[42]:


# Voting Classifier with weights

vc1 = VotingClassifier(estimators=[('logreg',logreg),('dectree',dectree),('ranfor',ranfor),('knn',knn),('abc',abc)], 
                      voting='soft', weights=[2,1,2,2,1])
vc1.fit(X_train, y_train)


# In[43]:


y_pred_vc1 = vc1.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc1))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc1))


# In[50]:


print('Model Accuracy')
print('\n')
print('Logistic Regression: '+str(round(accuracy_score(y_test, y_pred_logreg)*100,2))+'%')
print('Decision Tree: '+str(round(accuracy_score(y_test, y_pred_dectree)*100,2))+'%')
print('KNN: '+str(round(accuracy_score(y_test, y_pred_knn)*100,2))+'%')
print('\n')
print('Stacking method')
print("Stacking Accuracy:", accuracy_score(y_test, stacking_pred))
print('\n')
print('Averaging Method')
print('Random Forest: '+str(round(accuracy_score(y_test, y_pred_ranfor)*100,2))+'%')
print('\n')
print('Boosting Method')
print('AdaBoost: '+str(round(accuracy_score(y_test, y_pred_abc)*100,2))+'%')
print('\n')
print('Voting Classifiers')
print('Voting Classifier without Weights: '+str(round(accuracy_score(y_test, y_pred_vc)*100,2))+'%')
print('Voting Classifier with Weights: '+str(round(accuracy_score(y_test, y_pred_vc1)*100,2))+'%')


# # We see slight improvement in the accuracy with weights assigned to the models.

# In[47]:


# Stacking
from sklearn.ensemble import StackingClassifier
stacking_model = StackingClassifier(estimators=[('dt', DecisionTreeClassifier())], 
                                    final_estimator=DecisionTreeClassifier())
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_test)


# In[49]:


print("Stacking Accuracy:", accuracy_score(y_test, stacking_pred))


# In[ ]:





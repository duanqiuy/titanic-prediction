
# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization(for EDA)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

#We will use the popular scikit-learn library to develop our machine learning algorithms

# Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc

# Models
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

import missingno as msno
import string

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df_test = pd.read_csv("/Users/qiuyingduan/Desktop/MMAI/5000/5000Assignment2/titanic/test.csv")
df_train = pd.read_csv("/Users/qiuyingduan/Desktop/MMAI/5000/5000Assignment2/titanic/train.csv")
def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
def divide_df(all_data):
    # Use DataFrame.loc attribute to access a particular cell in the given Dataframe using the index and column labels.
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
    # Returns divided dfs of training and test set 
df_all = concat_df(df_train, df_test)
df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 
dfs = [df_train, df_test]  # List consisting of both Train and Test set


# In[3]:


# A little bit taste of the data set
df_all.sample(10)


# In[4]:


#information of the data set
df_all.info()


# In[5]:


# Information of the training set
df_train.info()


# In[6]:


df_train.sample(10)


# In[7]:


#Information of the test set
df_test.info()


# In[8]:


df_test.sample(10)


# In[9]:


df_train.describe() 


# In[10]:


#data set miss ages ,cabins and embarked values
df_train.isnull().sum()


# In[11]:


df_test.isnull().sum()


# In[12]:


#Vistuation of the missing values in training set
msno.matrix(df_train,figsize=(12,5))# missing values


# In[13]:


#Visulization of the missing value in test dataset
msno.matrix(df_test,figsize=(12,5))


# In[14]:


#Check the survival Situation 
total_survived= df_train['Survived'].sum()# Survival Situation
total_no_survived = 891 - total_survived

plt.figure(figsize = (10,5))
plt.subplot(121)#first pic
sns.countplot(x='Survived', data=df_train)
plt.xlabel("Survived Distribution")
plt.title('Survival Count')

plt.subplot(122)#second pic
plt.pie([total_no_survived, total_survived],labels=['No Survived','Survived'],autopct='%1.2f%%')
plt.title('Survival Rate') 

plt.show()


# In[15]:


#The Relationship between Single Attributes and Survival Rate


# In[16]:


# Number of people in different Pclass
df_train[['Pclass','Name']].groupby(['Pclass']).count()


# In[17]:


# Visualzation of Number of people in different Pclass 1
df_train[['Pclass','Name']].groupby(['Pclass']).count()
plt.figure(figsize= (10 ,5))
sns.countplot(x='Pclass', data=df_train)
plt.title('Person Count Across on Pclass')
plt.show()


# In[18]:


# Visualzation of Number of people in different Pclass 1
plt.figure(figsize= (10 ,5))
plt.pie(df_train[['Pclass','Name']].groupby(['Pclass']).count(),        labels=['1','2','3'],autopct='%1.0f%%')
plt.axis("equal")#pie chart

plt.show()


# In[19]:


#Survival rate in different Pclass
df_train.pivot_table(values="Survived",index="Pclass",aggfunc=np.mean)


# In[20]:


plt.figure(figsize= (10 ,5))
sns.barplot(data=df_train,x="Pclass",y="Survived",ci=None)#ci表示置信区间

plt.show()


# In[21]:


# Female and Male Passenger Distribution
df_train[['Sex','Name']].groupby(['Sex']).count()


# In[22]:


#The Relationship between Sex and Survival Rate
df_train.pivot_table(values="Survived",index="Sex",aggfunc=np.mean)


# In[23]:


plt.figure(figsize= (8 ,5))# Sex and Survival Rate
sns.barplot(data=df_train,x="Sex",y="Survived",ci=None)

plt.show()


# In[24]:


# Number of people in each Embarked Spot
df_train[['Embarked','Name']].groupby(['Embarked']).count()


# In[25]:


#Relationship between Embarked and the Survival rate 
sns.factorplot('Embarked','Survived', data=df_train,size=5,aspect=2)
plt.show()


# In[26]:


#Age Range Separation
df_train["AgeRange"]=pd.cut(df_train["Age"],10)
df_train.AgeRange.value_counts(sort=False)#number of people in each age range


# In[27]:


df_train.pivot_table(values="Survived",index="AgeRange",aggfunc=np.mean)


# In[28]:


#Relationship between SibSp and Survival Rate, Parch and Survival Rate
#Combine SibSp and Parch attributes
data1=df_train.copy() 
data2=df_test.copy() 
data1['Family_size'] = data1['SibSp'] + data1['Parch'] +1
data1['Family_size'].value_counts().sort_values(ascending=False)
sns.factorplot('Family_size','Survived', data=data1,size=5,aspect=2)
plt.show()


# In[29]:


#Relationship between Fare value and Survival Rate
df_train["FareRange"]=pd.cut(df_train["Fare"],5)#Age Range Separation
df_train.FareRange.value_counts(sort=False)


# In[30]:


df_train.pivot_table(values="Survived",index="FareRange",aggfunc=np.mean)# Higher Fare, higher survival rate


# In[31]:


#Dealing with Missing Values 
#Age has missing values, median ages in different Pclass is used to fill the missing value
age_median_psex = df_train.groupby(["Pclass","Sex"]).Age.median()
print(age_median_psex)


# In[32]:


# Fill the training set missing age values
df_train.set_index(["Pclass","Sex"],inplace=True)

df_train.Age.fillna(age_median_psex,inplace=True)#fillna

df_train.reset_index(inplace=True)


# In[33]:


# Fill the testing set missing age values
df_test.set_index(["Pclass","Sex"],inplace=True)

df_test.Age.fillna(age_median_psex,inplace=True)#fillna

df_test.reset_index(inplace=True)


# In[34]:


# Most people embarked from S; Pclass3 C/Q are similar; Pclass1/2 C>Q    
df_train.groupby(["Pclass","Embarked"]).Name.count()


# In[35]:


#Put'S'in Embarked missing value
df_train.fillna({"Embarked":"S"},inplace=True)


# In[36]:


#Fill the missing Fare value in test set
df_test[df_test['Fare'].isnull()]
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]
df_test['Fare'] = df_test['Fare'].fillna(med_fare) 


# In[ ]:





# In[37]:


#The relationship of Sex, AgeRange, and Survival Rate
plt.figure(figsize= (10 ,5))
sns.pointplot(data=df_train,x="AgeRange",y="Survived",hue="Sex",ci=None,
             markers=["^", "o"], linestyles=["-", "--"])
plt.xticks(rotation=60)

plt.show()


# In[38]:


#female have higher survival rate and it increases with the increase of age range while male has lower survival rate.
#Most survival male passenger is (0-16)years old


# In[39]:


#The Relationship of Sex Age Pclass and Survival Rate
df_train.pivot_table(values="Survived",index="AgeRange",columns=["Sex","Pclass"],aggfunc=np.mean)


# In[40]:


sns.FacetGrid(data=df_train,row="AgeRange",aspect=2.5).map(sns.pointplot,"Pclass","Survived","Sex",hue_order=["male","female"],ci=None,palette="deep", 
     markers=["^", "o"], linestyles=["-", "--"]).add_legend()

plt.show()


# In[41]:


#The relationship of sex age embarked Pclass and survival rate
df_train.pivot_table(values="Survived",index="Sex",columns=["Pclass","Embarked"],aggfunc=np.mean)


# In[42]:


sns.FacetGrid(data=df_train,row="Embarked",aspect=2.5).map(sns.pointplot,"Pclass","Survived","Sex",hue_order=["male","female"],ci=None,palette="deep", 
     markers=["^", "o"], linestyles=["-", "--"]).add_legend()

plt.show()


# In[43]:


#Feature transformation
#Transfer Age---Passengers are under 16 and between 32 and 56 has higher survival rate
#def under16(row):
    #result = 0.0
    #if row<16:
        #result = 1.0
    #return result
#def between32and56(row):
    #result = 0.0
    #if row>=32 and row<56:
        #result = 1.0
    #return result
#df_train['under16'] = df_train['Age'].apply(under16)
#df_train['between32and56'] = df_train['Age'].apply(between32and56)
#df_test['under16'] = df_test['Age'].apply(under16)
#df_test['between32and56'] = df_test['Age'].apply(between32and56)

#Drop Age Attribute
df_train.drop('AgeRange',axis=1,inplace=True)

#df_train.drop('Age',axis=1,inplace=True)
#df_test.drop('Age',axis=1,inplace=True)

# Transfer Fare Range---Fare Price is under 102
#def farelessthan102(row):
    #result = 0.0
    #if row<102:
        #result = 1.0
    #return result

#df_train['farelessthan102'] = df_train['Fare'].apply(farelessthan102)
#df_test['farelessthan102'] = df_test['Fare'].apply(farelessthan102)

#Drop Fare Attribute
df_train.drop('FareRange',axis=1,inplace=True)
#df_train.drop('Fare',axis=1,inplace=True)
#df_test.drop('Fare',axis=1,inplace=True)

# Transfer Family Size
df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch'] + 1
# Mapping Family Size
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_train['Family_Size_Grouped'] = df_train['Family_Size'].map(family_map)

df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch'] + 1
# Mapping Family Size
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_test['Family_Size_Grouped'] = df_test['Family_Size'].map(family_map)

df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
# Mapping Family Size
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)


# In[44]:


df_test.info()


# In[45]:


df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



df_train['Title'] = df_train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_train['Is_Married'] = 0
df_train['Is_Married'].loc[df_train['Title'] == 'Mrs'] = 1
df_train['Title'] = df_train['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_train['Title'] = df_train['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


df_test['Title'] = df_test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_test['Is_Married'] = 0
df_test['Is_Married'].loc[df_test['Title'] == 'Mrs'] = 1
df_test['Title'] = df_test['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_test['Title'] = df_test['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')







# In[46]:


#Check the correlation
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[47]:


df_train.info()


# In[48]:


onehot_features = ['Sex', 'Embarked', 'Title', 'Family_Size_Grouped','Pclass']
encoded_features = []

for df in dfs:
    for feature in onehot_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

# *encoded_features gives all encoded features of each of Six onehot_features         
df_train = pd.concat([df_train, *encoded_features[:5]], axis=1)
df_test = pd.concat([df_test, *encoded_features[5:]], axis=1)


# In[49]:



drop_cols = ['Embarked', 'Family_Size', 'Family_Size_Grouped','PassengerId',
             'Ticket','Sex', 'Cabin','SibSp','Parch','Name','Pclass','Title']


# In[50]:


df_all.drop(columns=drop_cols, inplace=True)
df_train.drop(columns=drop_cols, inplace=True)
df_test.drop(columns=drop_cols, inplace=True)


# In[51]:


df_train.info()


# In[52]:


df_traincopy=df_train.copy()

X = df_train.drop('Survived',axis=1)
Y_train = df_traincopy['Survived'].values
X_train = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(df_test)


# In[53]:


random_forest = RandomForestClassifier(n_estimators=100)#Random Forest
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[54]:


logreg = LogisticRegression()#logistic regression
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# In[55]:


linear_svc = LinearSVC()#Linear Support Vector Machine
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


# In[56]:


decision_tree = DecisionTreeClassifier()#Decision Tree 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[57]:


# Here I struggle with should I change the age grouping and fare grouping, based on the model score, I chose not grouping
results = pd.DataFrame({
    'Model': ['Random Forest','Logistic Regression','Linear Support Vector Machines','Decision Tree'],
    'Score': [acc_random_forest, acc_log, acc_linear_svc, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[58]:


from sklearn.model_selection import cross_val_score#K-Fold Cross Validation 
rf = RandomForestClassifier(n_estimators=100,oob_score=True)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[59]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[60]:


importances = pd.DataFrame({'feature':X.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head()


# In[61]:


importances.plot.bar()


# In[62]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[63]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# In[64]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))


# In[65]:


#F1 Score 
from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


# In[66]:


#AUC curve
from sklearn.metrics import roc_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# In[67]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[68]:


submission = pd.DataFrame({
        "PassengerId": data2["PassengerId"],
        "Survived": Y_prediction
    })

submission.to_csv('submission.csv', index=False)

data=pd.read_csv("submission.csv")
data.head(10)


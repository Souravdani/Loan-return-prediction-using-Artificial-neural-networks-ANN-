# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:51:19 2022
Loan return prediction using Artificial neural networks (ANN)
@author: Soura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("C:\\Users\\Soura\\Downloads\\lending_club_loan_two.csv")
df.columns
df.info()

# We are attempting to predict loan_status, so creating a countplot.
sns.countplot(x='loan_status',data=df)

# Histogram of the loan_amnt column.
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)

# Now exploring correlation between the continuous feature variables.
df.corr()

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)
## We can see that the installment has nearly perfect correlation with loan_amnt
# so, lets explore that further

sns.scatterplot(x='installment',y='loan_amnt',data=df,)

sns.boxplot(x='loan_status',y='loan_amnt',data=df)
df.groupby('loan_status')['loan_amnt'].describe()

sorted(df['grade'].unique()) # 7 unique features
sorted(df['sub_grade'].unique())

## countplot per grade setting the hue to the loan_status label
#count plot per subgrade
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,
              palette='coolwarm' ,hue='loan_status')
## It looks like F and G subgrades don't get paid back often.
# Isloating those and recreating the countplot for those subgrades.


########## Data PreProcessing ###########

len(df)  # 396030
df.isnull().sum()

## Converting these missing values in terms of percentage of the total dataframe:
100* df.isnull().sum()/len(df)

df['emp_title'].nunique()  #173105
df['emp_title'].value_counts()
## There are too many unique job titles to convert into dummy variable feature. 
# So, removing that emp_title column.
df = df.drop('emp_title',axis=1)

df['emp_length'].dropna().unique()

## Charge off rates are extremely similar across all employment lengths. 
# So, dropping the emp_length column.
df = df.drop('emp_length',axis=1)

## Reviewing the title column vs the purpose column. If there is repeated information?
df['purpose'].head(10)
df['title'].head(10)

## Title column is simply a string subcategory/description of the purpose column.
# So, going ahead and dropping the title column:
df = df.drop('title',axis=1)

## Creating a value_counts of the mort_acc column.
df['mort_acc'].value_counts()

## Let's review the other columns to see which most highly correlates to mort_acc
df.corr()['mort_acc'].sort_values()

## Looks like the total_acc feature correlates with mort_acc, and this makes sense.
# so, trying fillna() method. We will group the dataframe by the total_acc 
# and calculate the mean value for the mort_acc per total_acc entry. 

df.groupby('total_acc').mean()['mort_acc']

## Now filling in the missing mort_acc values based on their total_acc value. 
# If the mort_acc is missing, then we will fill that missing value with the 
# mean value corresponding to its total_acc value from the series we have created above.

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg[2.0]


def fill_mort_acc(total_acc,mort_acc):
    
    '''Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.'''
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df.isnull().sum()

## revol_util and the pub_rec_bankruptcies have missing data points, 
# but they account for less than 0.5% of the total data. 
# So removing the rows that are missing those values in those columns with dropna().
df = df.dropna()
df.isnull().sum()  ## We are clear of na values



########### Categorical Variables and Dummy Variables ################


## Listing all the columns that are currently non-numeric:
df.select_dtypes(['object']).columns

## Converting the term feature into integer numeric data type using .apply() funcn
df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))

## We also know grade is part of sub_grade, so dropping the grade feature:
df = df.drop('grade',axis=1)

## Now converting the subgrade into dummy variables. 
# Then concatenating these new columns to the original dataframe. 
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns
df.select_dtypes(['object']).columns

## Converting the columns: 
# ['verification_status', 'application_type','initial_list_status','purpose'] 
# into dummy variables and then concatenating them with the original dataframe

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df['home_ownership'].value_counts()

## Converting these to dummy variables, but replace NONE and ANY with OTHER,  
# so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. 
# Then concatenate them with the original dataframe

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

## Let's feature engineer a zip code column from the address in the data set. 
# Creating column 'zip_code' that extracts the zip code from the address column.

df['zip_code'] = df['address'].apply(lambda address:address[-5:])

## Now making this zip_code column into dummy variables using pandas. 
# and concatenating the result and dropping the original zip_code column 
## along with dropping the address column.

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

## we wouldn't know beforehand whether or not a loan would be issued when using 
# our model, so we won't have an issue_date.

df = df.drop('issue_d',axis=1)

## Extracting the year from the feature 'earliest_cr_line' using a .apply function, 
# then converting it to a numeric feature. 
# Setting this new data to a feature column called 'earliest_cr_year'.
# Then dropping the earliest_cr_line feature.

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)
df.select_dtypes(['object']).columns




############## Train Test Split ###############



from sklearn.model_selection import train_test_split

df = df.drop('loan_status',axis=1)
df.columns

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

######### Normalizing the Data #############

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


################### Creating the Model $################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


model = Sequential()

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )

from tensorflow.keras.models import load_model

model.save('full_data_project_model.h5')  


################### Evaluating Model Performance ############

from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))

confusion_matrix(y_test,predictions)


























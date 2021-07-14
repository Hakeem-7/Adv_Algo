import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Returns the current working directory
print os.getcwd()  


# Producing a cosine graph
x = np.linspace(0, 20, 100)
plt.plot(x, np.cos(x))
plt.title("Cosine Graph")
plt.show()

# Producing a sine graph
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.title("Sine Graph")
plt.show()




# Class examples and exercise
## Cleaning continous variables
titanic = pd.read_csv('C:/LinkedIn/Ex_Files_Machine_Learning_Algorithms/Exercise Files/titanic.csv')
titanic.head(10) # Displays the first 10 rows of the dataset
titanic.isnull().sum() # sum of the missing values in each column of the dataset
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True) # Since "Age" doesn't mean much, replace the missing values "na" with the average age of the passengers.
titanic.head(10)
## Evaluating and combining the "Simbling and Spouse = Sibsp" and "Parent and children = Parch" columns
for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )
#plt.show()

####################################################
import os
os.getcwd() #Returns our current working directory
####################################################

#Now, let's try to  change the working directory
os.chdir('C:/LinkedIn/Ex_Files_Machine_Learning_Algorithms/Exercise Files')
os.getcwd()

# Now let's proceed with our data cleaning
# Merge Sibsp and Parch
titanic['Family_count'] = titanic['SibSp'] + titanic['Parch']
# Drop unnecessary variables to eliminate multi-collinearity issues
titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis = 1, inplace = True) #axis=1 means drop column
titanic.head(10)

## Cleaning categorical variables
titanic.isnull().sum()
titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean() #mean was incorporated because the "Survived" variable is binary.
titanic['Cabin_status'] = np.where(titanic['Cabin'].isnull(), "missing", "found")
#titanic.drop(['Cabin_ind'], axis = 1, inplace = True)
titanic.head()

# Convert "Sex" variable to numeric - Mapping technique
gender_num = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(gender_num) 
titanic.head()

# Drop unnecessary variables
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
titanic.head()
# Write out the cleaned data
titanic.to_csv('titanic_ak_cleaned.csv', index = False) # "Index = False" removes the index column


## Foundations: Split data into Train, Validation, and Test set
from sklearn.model_selection import train_test_split

titanic_c = pd.read_csv('titanic_ak_cleaned.csv')
titanic.c.head()
features = titanic_c.drop('Survived', axis = 1)
label = titanic_c['Survived']
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state=42)

for dataset in [y_train, y_val, y_test]:
    print(round(len(dataset) / len(label), 2)) # confirms that the split was properly done

# Write out all the data
x_train.to_csv('train_ak_features.csv', index=False)
x_val.to_csv('val_ak_features.csv', index=False)
x_test.to_csv('test_ak_features.csv', index=False)

y_train.to_csv('train_ak_label.csv', index=False)
y_val.to_csv('val_ak_label.csv', index=False)
y_test.to_csv('test_ak_label.csv', index=False)

## Logistic Regression
import joblib #for saving implemented algos
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV #Cross-Validation for hyperparameter tuning.
import warnings #Ignore warnings --Deprecations
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

#Now, let's try to  change the working directory
# https://marketplace.visualstudio.com/items?itemName=ionutvmi.path-autocomplete
import os
os.chdir('C:/LinkedIn')
os.getcwd()

tr_features = pd.read_csv('./Ex_Files_Machine_Learning_Algorithms/Exercise Files/train_features.csv')
tr_labels = pd.read_csv('./Ex_Files_Machine_Learning_Algorithms/Exercise Files/train_labels.csv')









## Version control coding - Quick Digression!
#git remote add origin https://github.com/Hakeem-7/Adv_Algo.git
#PS C:\LinkedIn> git add .
#PS C:\LinkedIn> git commit -m "first commit"
# PS C:\LinkedIn> git push -u origin master


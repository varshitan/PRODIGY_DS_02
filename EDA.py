import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data Cleaning
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])

# EDA
print(train_data.describe())

# Correlation Heatmap
numeric_columns = train_data.select_dtypes(include=[np.number])
corr_matrix_train = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_train, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Train)')
plt.show()

# Graph 1: Survival Pie Chart
survival_counts = train_data['Survived'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(survival_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90)
plt.title('Survival Percentage')
plt.show()

# Graph 2: Passenger Class Distribution
sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# Graph 3: Gender Distribution as a Pie Chart
gender_counts = train_data['Sex_female'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.show()

# Graph 4: Age Distribution as a Histogram
age_values = train_data['Age'].values
plt.figure(figsize=(8, 6))
sns.histplot(age_values, bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Graph 5: Sibling/Spouse (SibSp) Distribution as a Box Plot
sns.boxplot(x='SibSp', y='Age', data=train_data)
plt.title('Age Distribution by SibSp')
plt.xlabel('SibSp')
plt.ylabel('Age')
plt.show()

# Graph 6: Parent/Children (Parch) Distribution as a Box Plot
sns.boxplot(x='Parch', y='Age', data=train_data)
plt.title('Age Distribution by Parch')
plt.xlabel('Parch')
plt.ylabel('Age')
plt.show()

# Graph 7: Fare Distribution as a Histogram
fare_values = train_data['Fare'].values
plt.figure(figsize=(8, 6))
sns.histplot(fare_values, bins=20, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Graph 8: Embarked Distribution as a Count Plot
sns.countplot(x='Embarked', data=train_data)
plt.title('Embarked Distribution')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()



survival_by_class = train_data.groupby('Pclass')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))
sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
plt.title('Survival Percentage by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Percentage')
plt.ylim(0, 100)
plt.show()

print(survival_by_class)

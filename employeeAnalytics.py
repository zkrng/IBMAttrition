"""
IBM is an American MNC operating in around 170 countries with major business vertical as computing, software, 
and hardware. Attrition is a major risk to service-providing organizations where trained and experienced people are 
the assets of the company. The organization would like to identify the factors which influence the attrition of 
employees. 

Data Dictionary

Age: Age of employee
Attrition: Employee attrition status
Department: Department of work
DistanceFromHome
Education: 1-Below College; 2- College; 3-Bachelor; 4-Master; 5-Doctor;
EducationField
EnvironmentSatisfaction: 1-Low; 2-Medium; 3-High; 4-Very High;
JobSatisfaction: 1-Low; 2-Medium; 3-High; 4-Very High;
MaritalStatus
MonthlyIncome
NumCompaniesWorked: Number of companies worked prior to IBM
WorkLifeBalance: 1-Bad; 2-Good; 3-Better; 4-Best;
YearsAtCompany: Current years of service in IBM
Analysis Task:
- Import attrition dataset and import libraries such as pandas, matplotlib.pyplot, numpy, and seaborn.
- Exploratory data analysis

Find the age distribution of employees in IBM
Explore attrition by age
Explore data for Left employees
Find out the distribution of employees by the education field
Give a bar chart for the number of married and unmarried employees
- Build up a logistic regression model to predict which employees are likely to attrite.
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
from sklearn.linear_model import LogisticRegression

results_path = r'.\results'
if not os.path.exists(results_path):
    os.mkdir(results_path)

df = pd.read_csv('IBM Attrition Data.csv')
# print(df.info())
# print(df.describe())
#
# for c in df.columns:
#     if c in ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'YearsAtCompany']:
#         df.boxplot(column=[c], by=['Attrition'], notch=True)
#         plt.savefig(f'.\\results\\Attrition_{c}_boxplot.png')
#     else:
#         df2 = pd.crosstab(index=df[c], columns=df['Attrition'])
#         df2.plot.bar()
#         plt.savefig(f'.\\results\\Attrition_{c}_boxplot.png')

x = df.drop(columns=['Attrition', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'EducationField', 'Department', 'MaritalStatus'])
x_dept = pd.get_dummies(df['Department'], drop_first=True)
x_marital = pd.get_dummies(df['MaritalStatus'], drop_first=True)
x_dept = x_dept.join(x_marital, how='right')
x = x.join(x_dept, how='right')
x.rename(columns={'Research & Development':'RnD'}, inplace=True)
# print(x.head())
y = df['Attrition']
from sklearn.decomposition import PCA
random_state = 2
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
pca = PCA(n_components=2).fit(x)
print(f'{pca.explained_variance_ratio_}')
x_pca = pca.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

dtree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
print(accuracy_score(y_pred=y_pred, y_true=y_test))

# plt.figure(figsize=(200, 50), dpi=300)
# tree.plot_tree(dtree, feature_names=x.columns)
# plt.savefig(f'.\\results\\DecisionTree.png')

# from sklearn.inspection import DecisionBoundaryDisplay
# DecisionBoundaryDisplay.from_estimator(dtree, x_pca, cmap=plt.cm.RdYlBu, response_method='predict')

# explore the data
import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=x.columns,
                                class_names=y.unique(),
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Attrition Decision Tree")
graph.view(filename='.\\results\\Attrition Decision Tree_graph.pdf')
# graph.save(filename='.\\results\\Attrition Decision Tree_graph.pdf')

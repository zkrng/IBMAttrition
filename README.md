# IBM Attrition HR analytics 

The goal of this project is to identify which factors might influence the attrition of IBM employees. The code works on IBM_Attrition_Data.csv dataset provided in the course [Data Science with Python][course]. The almost similar dataset can be found in [Kaggle hub][data].

## Notes
 - Dataset description (provided by the course) is mentioned in [employeeAnalytics.py][description]
 - [employeeAnalytics.py][description] also provides descriptive statistics and visualization of each attribute relating to **Attrition** attribute

## Run the code
Compare ensemble learning methods by running:
```sh
python ensembleLearning.py
```
As by 2022-12-15, the highest accuracy score (0.851) is achieved with RandomForestClassifier and ExtraTreeClassifier on 8 components derived from PCA. This score is higher than the solution provided by the course (0.844), which utilized LogisticRegression on all 9 attributes.

[//]:#
   [course]: <https://lms.simplilearn.com/courses/2772/Data-Science-with-Python/syllabus>
   [data]: <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
   [description]: <./employeeAnalytics.py>

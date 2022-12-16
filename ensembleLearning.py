import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# Load data
df = pd.read_csv('IBM Attrition Data.csv')
x = df.drop(columns=['Attrition', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'EducationField', 'Department', 'MaritalStatus'])
x_dept = pd.get_dummies(df['Department'], drop_first=True)
x_marital = pd.get_dummies(df['MaritalStatus'], drop_first=True)
x_dept = x_dept.join(x_marital, how='right')
x = x.join(x_dept, how='right')
x.rename(columns={'Research & Development':'RnD'}, inplace=True)
# print(x.head())
y_orig = df['Attrition'].replace({'Yes': 1, 'No': 0}).to_numpy()
# x_orig = x[['MonthlyIncome', 'Age', 'YearsAtCompany']]

random_state = 2
# x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2, random_state=2)

# Standardize
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std
x = x.to_numpy()

# Parameters
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

plot_idx = 1

models = [
    GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, subsample=0.5, learning_rate=0.01,
                               min_samples_leaf=1, random_state=random_state),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=n_estimators),
    ExtraTreesClassifier(n_estimators=n_estimators),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators),
    HistGradientBoostingClassifier()
]

acc_score = np.ndarray(shape=(6, 7))
ncomponent_range = np.arange(3, 10)
# PCA
for n_components in ncomponent_range:
    print(30*'=')
    # Shuffle
    idx = np.arange(x.shape[0])
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idx)
    XX = x[idx]
    Y = y_orig[idx]
    pca1 = PCA(n_components=n_components).fit(XX)
    print(n_components)
    print(f'{pca1.explained_variance_ratio_}')

    x_pca = pca1.transform(XX)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, Y, test_size=0.25, random_state=random_state)

    # x_train = x_train.to_numpy()
    # y = y_train.to_numpy()
    # x_test = x_test.to_numpy()
    kk = 0
    for model in models:
        # We only take the two corresponding features
        X = x_train.copy()
        y = y_train.copy()

        # Train
        model.fit(X, y)

        scores = model.score(x_test, y_test)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][: -len("Classifier")]

        model_details = model_title
        print(scores)

        y_pred = model.predict(x_test)
        ac = accuracy_score(y_true=y_test, y_pred=y_pred)
        acc_score[kk, n_components-3] = ac

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        kk+=1

max_score_loc = np.where(acc_score==acc_score.max())
(loc1, loc2) = max_score_loc
print(f'Highest accuracy score: {acc_score.max()} by models {loc1} with {loc2+3}  components ')
print(ncomponent_range)
print(acc_score)


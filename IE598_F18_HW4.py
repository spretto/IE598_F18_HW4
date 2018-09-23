import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets.samples_generator import make_regression

X, y, cf= make_regression(n_samples=1000, n_features=100, noise=0, coef=True, random_state=42)



def plot_residuals(name, test_pred, train_pred):
    plt.scatter(train_pred, train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(test_pred, test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel(name + ' Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()

    
def print_r2_mse_score(name, model, model_test_pred, model_train_pred):
    print(name + " score on test: ", model.score(X_test, y_test))
    print('Slope: %.3f' % model.coef_[0])
    print("Model Coefficients", model.coef_)
    print('Intercept: %.3f' % model.intercept_)
    print('MSE train: %.3f, test: %.3f' % ( mean_squared_error(y_train, model_train_pred), mean_squared_error(y_test, model_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, model_train_pred), r2_score(y_test, model_test_pred)))
    
df = pd.read_csv('./housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
'NOX', 'RM', 'AGE', 'DIS', 'RAD',
'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

'''
#1. EDA PHASE HOUSING
print(df.head())
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
names = df.drop('MEDV', axis=1).columns
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

y = df['MEDV'].values
RM = df['RM'].values
plt.scatter(RM, y)
plt.xlabel("Number of Rooms")
plt.ylabel("Price in $1000s")
plt.show()
'''

#1. EDA PHASE 

df = pd.read_csv('./make_regressionX.csv', header=None, prefix='V')
print(df.head())

sns.pairplot(df, size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cf, xticklabels=cf
plt.show()

'''
y = df['MEDV'].values
RM = df['RM'].values
plt.scatter(RM, y)
plt.xlabel("Number of Rooms")
plt.ylabel("Price in $1000s")
plt.show()
'''

#2. TEST TRAIN SPLIT
X = df.iloc[:, :-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#3. FIT A LINEAR MODEL
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print_r2_mse_score("Linear", slr, y_test_pred, y_train_pred)
plot_residuals("Linear", y_test_pred, y_train_pred)


X_rooms = X[:5]
y=y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1, 1)
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)



lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_test_pred = lasso.predict(X_test)
lasso_train_pred = lasso.predict(X_train)

print_r2_mse_score("LASSO", lasso, lasso_test_pred, lasso_train_pred)
plot_residuals("LASSO", lasso_test_pred, lasso_train_pred)

lasso_coef = lasso.fit(X,y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()





ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_test_pred = ridge.predict(X_test)
ridge_train_pred = ridge.predict(X_train)
print_r2_mse_score("Ridge", ridge, ridge_test_pred, ridge_train_pred)
plot_residuals("Ridge", ridge_test_pred, ridge_train_pred)



elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet.fit(X_train, y_train)
elanet_test_pred = elanet.predict(X_test)
elanet_train_pred = elanet.predict(X_train)
print_r2_mse_score("elanet", elanet, elanet_test_pred, elanet_train_pred)
plot_residuals("elanet", elanet_test_pred, elanet_train_pred)


print("My name is Stephen Pretto")
print("My NetID is: spretto2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv('C:/Users/prath/Downloads/MLData/gld_price_data.csv')
print(gold_data.head())

correlation = gold_data.corr()

plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')

print(correlation['GLD'])
sns.histplot(gold_data['GLD'],color='gold')
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('G0LD Price')
plt.legend()
plt.show()

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
print(y_pred)

error_score = metrics.r2_score(Y_test, test_data_prediction)

print("R squared error : ", error_score)
print(regressor.score(X_test, Y_test))



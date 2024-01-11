from random import randint
LIMIT = 1000
COUNT = 100
INPUT = list()
OUTPUT = list()
for i in range(COUNT):
	a = randint(0, LIMIT)
	b = randint(0, LIMIT)
	c = randint(0, LIMIT)
	op = a + (2 * b) + (3 * c)
	INPUT.append([a, b, c])
	OUTPUT.append(op)
	OUTPUT[:10]
	INPUT[:10]
from sklearn.linear_model import LinearRegression

predictor = LinearRegression(n_jobs =-1)

predictor.fit(X = INPUT, y = OUTPUT)

X_TEST = [[ 10, 20, 30 ]] 

outcome = predictor.predict(X = X_TEST)

coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))
from sklearn.neural_network import MLPClassifier
import numpy as np

def sklearn_NN_XOR():

	VT = 100000 #número de muestras de entrada
	E1 = np.random.randint(2, size=VT)
	E2 = np.random.randint(2, size=VT)
	SE = E1 ^ E2 # output función AND
	input = np.column_stack((E1, E2))
# Creamos las entradas y salidas de validación E1V, E2V, SEV para test
	vtest = 20 #número de muestras de entrada
	E1V = np.random.randint(2, size=vtest)
	E2V = np.random.randint(2, size=vtest)
	SEV = E1V ^ E2V
	input_test = np.column_stack((E1V, E2V))


  # Definimos el perceptrón MLPClassifier
#	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	clf = MLPClassifier(learning_rate_init=0.7,activation='logistic',hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(input, SE)
	sol=clf.predict(input_test)

	print("=======Solución======")
	print(SEV)
	print("=======Predicción=====")
	print(sol)

sklearn_NN_XOR()

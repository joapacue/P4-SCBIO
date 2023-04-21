import pdb
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def myperceptron_AND_23826389E():
# Función principal que realiza las funciones de:
# 1) Creación de las variables para el banco de entrenamiento y banco de validación
# 2) Creación de la red neuronal (perceptrón)
# 3) Entrenamiento de la red neuronal con el set de valores del banco de entrenamiento
# 4) Validación de la red con el banco de validación
# 5) Cálculo y representación del error cometido
# Creamos las entradas y salidas de entrenamiento para una función AND
# Inicializamos las variables E1, E2 y SE ideales para entrenamiento

	VT = 100000 #número de muestras de entrada
	E1 = np.random.randint(2, size=VT)
	E2 = np.random.randint(2, size=VT)
	SE = E1 & E2 # output función AND
	input = np.column_stack((E1, E2))
# Creamos las entradas y salidas de validación E1V, E2V, SEV para test
	vtest = 20 #número de muestras de entrada
	E1V = np.random.randint(2, size=VT)
	E2V = np.random.randint(2, size=VT)
	SEV = E1V & E2V
	input_test = np.column_stack((E1V, E2V))

	# Inicializamos un perceptrón para 2 entradas
	myperceptron = initialize_perceptron(2)

	# Entrenamos el perceptron para un LR, por defecto 0.7
	LR = 0.7
	myperceptronT=train_perceptron(myperceptron,LR,input,SE)

	# Evaluamos el perceptrón
	S_est=useperceptron(myperceptronT,input_test)
	error = np.mean(abs(SEV-S_est))

	# Visualización del Resultado
	### INCLUYE TU CÓDIGO para visualizar los datos de Test correctos y la solución
	# dada por el perceptrón
	return None

def initialize_perceptron(n_inputs):
# Esta función crea e inicializa la estructura myperceptron.weights
# donde se guardan los pesos del perceptron. Los valores de los pesos
# iniciales deben ser números aleatorios entre -1 y 1.
# n_inputs: numero de entradas al perceptron
# OUTPUTS
# bias: bias del perceptron
# weights: pesos del perceptron
	class perceptron:
		neurons=1
		bias = 1
		weights = np.random.rand(n_inputs+1)-np.random.rand(n_inputs+1)
	myperceptron = perceptron()
	return myperceptron

def sigmoid(x):
# Funcion de activacion sigmoide
	out= 1/(1+math.exp(-x))
	return out

def train_perceptron(myperceptron, LR, input, output):
# Función que modifica los pesos del perceptrón para que vaya aprendiendo a partir
# de los vales de entrada que se le indican
# INPUTS
# myperceptron: estructura con el perceptron
# LR: tasa de aprendizaje
# input: matriz con valores de entrada de entrenamiento ([E1 E2])
# output: vector con valores de salida de entrenamiento ([SE])
# OUTPUTS
# myperceptron: perceptron ya entrenado
# bias: bias del perceptron
# weights: pesos del perceptron
# ESTE PERCEPTRÓN UTILIZA:

# Función de activación sigmoidal
	res=0
	iter=0
	for x in input:
		xp=np.concatenate(([myperceptron.bias],x))
		res = np.cumsum(myperceptron.weights * xp)
		res=res[len(res)-1]
		res=sigmoid(res)
		if res!=output[iter]:
			correct=(LR*xp*(output[iter]-res))
			myperceptron.weights=myperceptron.weights+correct
		iter=iter+1
		print(iter)
		print(res)
		print(output[iter])
	return myperceptron
def useperceptron(myperceptron, input):
# funcion que utiliza el perceptron para calcular las salidas a partir de
# las entradas de acuerdo con lo que haya aprendido el perceptron en la
# fase de entrenamiento
# INPUTS
# myperceptron: perceptron
# input: entrada que se le pasara al perceptron (datos test)
# OUTPUTS
# out: salida
	res=0
	for x in input:
		xp=np.concatenate(([myperceptron.bias],x))
		res = np.cumsum(myperceptron.weights * xp)
		res=res[len(res)-1]
		res=sigmoid(res)
		#print(res)
	return res

myperceptron_AND_23826389E()

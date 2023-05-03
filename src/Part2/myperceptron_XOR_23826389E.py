import pdb
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def myperceptron_XOR_23826389E():
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
	SE = E1 ^ E2 # output función AND
	input = np.column_stack((E1, E2))
# Creamos las entradas y salidas de validación E1V, E2V, SEV para test
	vtest = 20 #número de muestras de entrada
	E1V = np.random.randint(2, size=vtest)
	E2V = np.random.randint(2, size=vtest)
	SEV = E1V ^ E2V
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
	ejex=np.arange(0,20,1)
	plt.scatter(ejex,SEV)
	plt.scatter(ejex,S_est,marker="x")
	plt.show()
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
		bias= 1
		weightsc1 = np.random.rand(2,n_inputs+1)-np.random.rand(2,n_inputs+1)
		weightsc2 = np.random.rand(1,3)-np.random.rand(1,3)
	myperceptron = perceptron()
	return myperceptron

def sigmoid(x):
# Funcion de activacion sigmoide
	#if abs(x)>708:
	#	if x<0:
	#		x=-708
	#	else:
	#		x=708
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
		xp = np.vstack((xp, xp))
		res = np.cumsum(myperceptron.weightsc1 * xp)
		res=res[[2,5]]
		res[0]=sigmoid(res[0])
		Y1=res[0]
		res[1]=sigmoid(res[1])
		Y2=res[1]

		res=np.cumsum(myperceptron.weightsc2 * np.concatenate(([myperceptron.bias],res)))
		res=res[len(res)-1]
		res=sigmoid(res)
		Y3=res
		
		if res!=output[iter]:
			errorc3=Y3*(1-Y3)*(output[iter]-Y3)
			correctc3=(LR*np.array([1, Y1, Y2])*errorc3)
			myperceptron.weightsc2=myperceptron.weightsc2+correctc3

			errorc1=Y1*(1-Y1)*errorc3*myperceptron.weightsc2[0][1]
			correctc1=(LR*np.concatenate(([myperceptron.bias],x))*errorc1)
			myperceptron.weightsc1[0]=myperceptron.weightsc1[0]+correctc1

			errorc2=Y2*(1-Y2)*errorc3*myperceptron.weightsc2[0][2]
			correctc2=(LR*np.concatenate(([myperceptron.bias],x))*errorc2)
			myperceptron.weightsc1[1]=myperceptron.weightsc1[0]+correctc2

		
		
		print(iter)
		print(res)
		print(output[iter])
		iter=iter+1
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
		xp = np.vstack((xp, xp))
		restemp = np.cumsum(myperceptron.weightsc1 * xp)
		restemp=restemp[[2,5]]
		restemp[0]=sigmoid(restemp[0])
		restemp[1]=sigmoid(restemp[1])

		restemp=np.cumsum(myperceptron.weightsc2 * np.concatenate(([myperceptron.bias],restemp)))
		restemp=restemp[len(restemp)-1]
		restemp=sigmoid(restemp)
		

		res=np.append(res, restemp)
	return res[1:len(res)]

myperceptron_XOR_23826389E()

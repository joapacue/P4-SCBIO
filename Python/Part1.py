
# Función principal que realiza las funciones de:
# 1) Creación de las variables para el banco de entrenamiento y banco de validación
# 2) Creación de la red neuronal (perceptrón)
# 3) Entrenamiento de la red neuronal con el set de valores del banco de entrenamiento
# 4) Validación de la red con el banco de validación
# 5) Cálculo y representación del error cometido

# Creamos las entradas y salidas de entrenamiento para una función AND

# Inicializamos las variables E1, E2 y SE ideales para entrenamiento 

# INCLUYE TU CÓDIGO

VT = 10000 #número de muestras de entrada
E1 = np.random.randint(2, size=VT)
E2 = np.random.randint(2, size=VT)
SE = E1 & E2 # output función AND
input = np.column_stack((E1, E2))


# Creamos las entradas y salidas de validación E1V, E2V, SEV para  test
vtest = 20 #número de muestras de entrada

E1V = np.zeros(2, size=vtest)
E2V = np.zeros(2, size=vtest)
SEV = E1V & E2V


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
print(S_est)



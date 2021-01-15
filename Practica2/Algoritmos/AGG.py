# enconding: utf-8

import numpy as np
import random
import math
import time
#import pandas as pd
#import os
#from os import listdir
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


NUM_MAX_GENERACIONES = 100000
k = 3
long_poblacion = 50
prob_cruce = 1
prob_muta = 0.001

coste_uniforme = False

class Cromosoma():
    def __init__(self,n,d):
        self.valores_asignados = np.zeros(n)
        self.centroides = np.zeros((k,d))
        self.vf_objetivo = 0.0


# Funcion que carga lo datos. Tanto el set de valores como la matriz de restricciones
# Se devuelve 4 parametros, la matriz de datos, la matriz de restricciones, y el numero de filas
# y columnas de nuestra matriz de datos

def cargar_datos(f_datos, f_restricciones):
    # Read data separated by commmas in a text file
    # with open("text.txt", "r") as filestream:

    f = open(f_datos, "r")
    f1 = open(f_restricciones, "r")
  
    """
    string1 = "datos/" + f_datos 
    string2 = "restricciones/" + f_restricciones

    f = open(string1, "r")
    f1 = open(string2, "r")
    """
   
   
    # MATRIZ DATOS 

    currentline_datos = []
    matrix_datos = []


    # print(line)
    # We need to change the input text into something we can work with (such an array)
    # currentline contains all the integers listed in the first line of "text.txt"
    # string.split(separator, maxsplit)
    # https://stackoverflow.com/questions/4319236/remove-the-newline-character-in-a-list-read-from-a-file

    for line in f: 
    #for line in f: 
        
        currentline_datos = line.rstrip('\n').split(",")

        matrix_datos.append(currentline_datos)

    
    # n == filas y d == columnas
    n = len(matrix_datos)
    d = len(currentline_datos)

    # MATRIZ RESTRICCIONES

    for i in range (n):
        for j in range (d):
            matrix_datos[i][j] = float(matrix_datos[i][j])


    currentline_const = []
    matrix_const = []

    for line in f1:
        # https://stackoverflow.com/questions/4319236/remove-the-newline-character-in-a-list-read-from-a-file
        currentline_const = line.rstrip("\n").split(",")

        matrix_const.append(currentline_const)


    return matrix_datos, matrix_const, n, d






# Funcion que calcula la infeasibility.
# Si en la matriz de restricciones marca un 1 y dos instancias NO estan en el mismo cluster
# o si en la matriz de restricciones hay un -1 y estan en el mismo, sumamos +1 el valor de la
# infeasibility ya que sera una RESTRICCION INCUMPLIDA


def calcular_infeasibility(datos, valores_asignados, restricciones):
    infeasability = 0

    for i in range(len(valores_asignados)):

        if valores_asignados[i] != 0:
            for j in range(i, len(valores_asignados)):
                if valores_asignados[j] != 0:
                    if restricciones[i][j] == '-1' and valores_asignados[i] == valores_asignados[j]:
                        infeasability = infeasability + 1
                    
                    elif restricciones[i][j] == '1' and valores_asignados[i] != valores_asignados[j]:
                        infeasability = infeasability + 1

    return infeasability
 


# Funcion que recalcula los centroides

def recalcular_centroides(datos, vector_asignados, centroides, d):

    for i in range(k):
        nuevo_centroide = np.zeros(d)
        num_total = 0

        for j in range (n):
            #Si este punto pertenece a este cluster
            if vector_asignados[j] == i + 1:
                for l in range (d):
                    nuevo_centroide[l] = nuevo_centroide[l] + datos[j][l]

            
                num_total = num_total + 1

        for j in range(d):
            if num_total != 0:
                nuevo_centroide[j] = nuevo_centroide[j]/num_total

        centroides[i] = nuevo_centroide




# Funcion que calcula cual es la distancia maxima de entre todas las instancias

def calcular_dmaxima(datos, n):
    maximo = float("-inf")
    dist = 0

    for i in range(n):
        for j in range(i+1, n):
            for c in range(d):
                dist = dist + math.pow((datos[i][c] - datos[j][c]),2)

            distancia = math.sqrt(dist)

            if(distancia > maximo):
                maximo = distancia

    return maximo



# Funcion que calcula el numero de restricciones de la matriz de restricciones

def calcular_nrestricciones(restricciones, n):
    num_restricciones = 0

    for i in range(n):
        for j in range(n):
            if (restricciones[i][j] == "1") or (restricciones[i][j] == "-1"):
                num_restricciones = num_restricciones + 1

    return num_restricciones



# Funcion que calcula el valor de lambda de la funcion objetivo 

def calcular_lambda(datos, restricciones, n):
    distancima_maxima = calcular_dmaxima(datos,n)

    num_restricciones = calcular_nrestricciones(restricciones, n)

    lamda = (int(distancima_maxima))/ num_restricciones

    return lamda


# Funcion que calcula la distancia

def distancia(datos, pos, centroides, d, vector_asignados):
    dif = 0
    contador = 0
   
    for j in range (len(datos)):
        if vector_asignados[j] == pos + 1:
            for i in range(d):
                dif = dif + math.pow((centroides[pos][i] - datos[j][i]),2)
            contador = contador + 1

    dist = math.sqrt(dif)

    return dist, contador



# Funcion para calcular la distancia media intra-cluster haciendo uso de la funcion "distancia"

def distancia_media(datos, pos, centroides, d, valores_asignados):
    dist, num_elem_dist = distancia(datos, pos, centroides, d, valores_asignados)

    distancia_media_ic = dist / num_elem_dist

    return distancia_media_ic


# Funcion para comprobar que ningun cluster queda vacio

def cluster_vacio(v_solucion, cluster):
    for i in range(len(v_solucion)):
        if v_solucion[i] == cluster:
            return 1

    return 0


def desviacion(datos, centroides, d, valores_asignados):
    desv = 0
    cont = 0

    for i in range(k):

        if (cluster_vacio(valores_asignados, i ) != 0):
            desv = desv + distancia_media(datos, i, centroides, d, valores_asignados)
            cont = cont + 1

    desviacion_general = desv / cont

    return desviacion_general




# FUNCION OBJETIVO DE LA BUSQUEDA LOCAL    

def f_objetivo(datos, valores_asignados, restricciones, n, d, centroides, valor_lambda):
    return desviacion(datos, centroides, d, valores_asignados) + (calcular_infeasibility(datos, valores_asignados, restricciones) * valor_lambda)



def generar_poblacion_aleatoria(long_poblacion,k,n,d):
    solucion = []

    for i in range(long_poblacion):
        cromo = Cromosoma(n,d)
        solucion.append(cromo)
        solucion[i].valores_asignados = np.copy(solucion[i].valores_asignados)

        for j in range(n):
            solucion[i].valores_asignados[j] = random.randint(1,k)      

    return solucion


def evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones):
    mejor_valor = float("inf")
    peor_valor = float("-inf")

    for i in range(long_poblacion):
        recalcular_centroides(datos, poblacion[i].valores_asignados, poblacion[i].centroides, d)

        poblacion[i].vf_objetivo = f_objetivo(datos, poblacion[i].valores_asignados, restricciones, n, d, poblacion[i].centroides, valor_lambda)
        num_evaluaciones = num_evaluaciones + 1

        if(poblacion[i].vf_objetivo < mejor_valor):
            mejor_valor = poblacion[i].vf_objetivo
            mejor_cromosoma = poblacion[i]
            pos_mejor = i

        if(poblacion[i].vf_objetivo > peor_valor):
            peor_valor = poblacion[i].vf_objetivo
            peor_cromosoma = poblacion[i]    
            pos_peor = i

    return mejor_cromosoma, peor_cromosoma,pos_mejor, pos_peor, num_evaluaciones


def seleccionar_poblacion(long_poblacion,poblacion):
    #Seleccion por torneo: se eligen aleatoriamente dos inidividuos de la poblacion 
    #y quedarse con el mejor de ellos
    #GENERACIONAL: se aplicaran tantos torneos como individuos existan en la poblacion
    #incluyendo los individuos ganadores

    num_torneos = 0
    nueva_poblacion = []
    #nueva_poblacion = np.zeros(long_poblacion,n)

    while(num_torneos < long_poblacion):
        aleatorio1 = random.randint(0,long_poblacion-1)
        aleatorio2 = random.randint(0,long_poblacion-1)

        if poblacion[aleatorio1].vf_objetivo < poblacion[aleatorio2].vf_objetivo:
            mejor_crom = poblacion[aleatorio1]
            nueva_poblacion.append(mejor_crom)
        else :
            mejor_crom = poblacion[aleatorio2]
            nueva_poblacion.append(mejor_crom)


        #nueva_poblacion[num_torneos] = np.array(np.copy(poblacion[mejor_crom]))
        num_torneos = num_torneos + 1

    return nueva_poblacion


def operador_uniforme(padre1, padre2, n_genes,d):
    # Generamos n/2 numeros aleatorios en el rango {0...n-1}
    aleatorios = np.zeros(int(n_genes/2))
    hijo = Cromosoma(n_genes,d)

    for i in range(int(n_genes/2)):
        aleatorios[i] = random.randint(0,(n_genes-1))
        
    hijo.valores_asignados = np.copy(padre2.valores_asignados)
    
    for i in aleatorios:
        hijo.valores_asignados[int(i)] = np.copy(padre1.valores_asignados[int(i)])

    return hijo


def segmento_fijo(padre1, padre2, n_genes,d):
    hijo = Cromosoma(n_genes,d)
    inicio_segmento = random.randint(0, (n_genes -1))
    tam_segmento = random.randint(0, (n_genes -1))
    num_elements = 0

    # Elegiremos al mejor padre, de manera que favoreceremos la explotacion
    if padre1.vf_objetivo < padre2.vf_objetivo:
        mejor_padre = padre1
        otro_padre = padre2
    else:
        mejor_padre = padre2
        otro_padre = padre1
    
    posicion = inicio_segmento

    hijo.valores_asignados = np.copy(otro_padre.valores_asignados)

    while num_elements < tam_segmento:
        hijo.valores_asignados[posicion] = np.copy(mejor_padre.valores_asignados[posicion])

        posicion = (posicion + 1)%n_genes
        num_elements = num_elements + 1

    restantes = n_genes - tam_segmento
    inicio = posicion

    #El resto del segmento se decidirá mediante coste uniforme
    aleatorios = np.zeros(int(restantes/2))

    for i in range(int(restantes/2)):
        aleatorios[i] = (random.randint(inicio,(inicio+restantes)))%n_genes
    
    for i in aleatorios:
        hijo.valores_asignados[int(i)] = np.copy(mejor_padre.valores_asignados[int(i)])


    return hijo
  


def reparacion(hijo, num_cluster,n_genes):
    posicion = random.randint(0,(n_genes-1))

    hijo.valores_asignados[posicion] = num_cluster

def cruce(num_esperado_cruces, poblacion, n_genes,d):
    #Operador cruce uniforme
    contador = 0
    individuo = 0

    p_intermedia = []
    #Generamos dos hijos
    while contador < num_esperado_cruces:
        if coste_uniforme == True:
            hijo = operador_uniforme(poblacion[individuo], poblacion[individuo+1], n_genes,d)
        else:
            hijo = segmento_fijo(poblacion[individuo], poblacion[individuo+1], n_genes,d)

        #reparar = True
        #while reparar:
            #reparar = False

        for i in range(k):
            if cluster_vacio(hijo.valores_asignados,i+1) == 0:
                #reparar = True
                reparacion(hijo,i+1,n_genes)
                break
        
        p_intermedia.append(hijo)

        if coste_uniforme == True:
            otro_hijo = operador_uniforme(poblacion[individuo], poblacion[individuo+1], n_genes,d)
        else:
            otro_hijo = segmento_fijo(poblacion[individuo], poblacion[individuo+1], n_genes,d)

        #reparar = True
        #while reparar:
            #reparar = False

        for i in range(k):
            if cluster_vacio(otro_hijo.valores_asignados,i+1) == 0:
                #reparar = True
                reparacion(otro_hijo,i+1,n_genes)
                break
        

        p_intermedia.append(otro_hijo)

        individuo = individuo+2
        contador = contador+1

    return p_intermedia


def mutacion(num_esperado_mutacion, poblacion ,n_genes,prob_muta, long_poblacion):
    #Calculo el cromosoma a realizar la primera mutacion
    mu_next = math.ceil(math.log(random.random()) / math.log(1.0 - prob_muta))
    contador = 0

    if prob_muta > 0:
        
        while contador < num_esperado_mutacion:
            #determinar cromosoma y gen a mutar
            i = ((int(mu_next/n_genes))%long_poblacion)
            j = (int(mu_next%n_genes))

            cluster_individuo = poblacion[i].valores_asignados[j]
            nuevo_valor = random.randint(1,k)

            poblacion[i].valores_asignados[j] = nuevo_valor

            while nuevo_valor == cluster_individuo:
                cluster_individuo = poblacion[i].valores_asignados[j]
                nuevo_valor = random.randint(1,k)
                poblacion[i].valores_asignados[j] = nuevo_valor
            
            
            #Se calcula la siguiente posicion a mutar
            if prob_muta < 1:
                mu_next = mu_next + math.ceil(math.log(random.random()) / math.log(1.0 - prob_muta))
            else:
                mu_next = mu_next + 1

            contador = contador + 1
            
        mu_next = mu_next - (n_genes * long_poblacion)

    return poblacion


def encontrar_cromosoma(poblacion, cromosoma):
    for i in range(len(poblacion)):
        if poblacion[i].vf_objetivo == cromosoma.vf_objetivo:
            if (poblacion[i].valores_asignados == cromosoma.valores_asignados).all():
                return True

    return False


def reemplazar_poblacion(poblacion_actual, poblacion_hijos, mejor_cromosoma, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones):
    # La poblacion de hijos sustituye automaticamente a la actual
    # Si la mejor sol. de la generacion anterior no sobrevive, sustituye 
    # directamente la peor de la poblacion

    poblacion_actual = np.copy(poblacion_hijos)

    #nuevo_mejor, nuevo_peor = evaluar_poblacion(poblacion_actual, long_poblacion, datos, restricciones, n, d, valor_lambda)

    if encontrar_cromosoma(poblacion_actual, mejor_cromosoma) == False:
        nuevo_mejor, nuevo_peor, pos_mejor, pos_peor, num_evaluaciones = evaluar_poblacion(poblacion_actual, long_poblacion, datos, restricciones, n, d, valor_lambda,num_evaluaciones)

        poblacion_actual[pos_peor].valores_asignados = np.copy(mejor_cromosoma.valores_asignados)
        poblacion_actual[pos_peor].vf_objetivo = np.copy(mejor_cromosoma.valores_asignados)
        poblacion_actual[pos_peor].centroides = np.copy(mejor_cromosoma.centroides)

    return poblacion_actual, num_evaluaciones



"""

ALGORITMOS

"""

# Algoritmo generacional etilista (AGG)

def AGG(datos, restricciones, n, d, long_poblacion, prob_cruce, prob_muta):
    num_evaluaciones = 0
    num_esperado_cruces = int(prob_cruce * (long_poblacion/2))
    num_esperado_mutacion = int(prob_muta * (long_poblacion*n))

    poblacion = generar_poblacion_aleatoria(long_poblacion, k, n, d)

    valor_lambda = calcular_lambda(datos, restricciones, n)

    #Evaluamos los individuos de la población inicial
    mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor , num_evaluaciones= evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)

    while(num_evaluaciones <= NUM_MAX_GENERACIONES):
        #print(num_evaluaciones)
        #Seleccionar P' desde P(t-1)
        nueva_poblacion = seleccionar_poblacion(long_poblacion,poblacion)
        p_cruzada = cruce(num_esperado_cruces, nueva_poblacion,n,d)
        p_hijos = mutacion(num_esperado_mutacion, p_cruzada,n,prob_muta,long_poblacion)

        poblacion, num_evaluaciones = reemplazar_poblacion(poblacion, p_hijos, mejor_cromosoma, long_poblacion,datos,restricciones,n,d,valor_lambda, num_evaluaciones)  
        mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor, num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)

    
    
    desviacion_general = desviacion(datos, mejor_cromosoma.centroides, d, mejor_cromosoma.valores_asignados)
    valor_inf = calcular_infeasibility(datos, mejor_cromosoma.valores_asignados, restricciones)
    valor_fobjetivo = f_objetivo(datos, mejor_cromosoma.valores_asignados, restricciones, n, d, mejor_cromosoma.centroides, valor_lambda)

    return mejor_cromosoma.valores_asignados, desviacion_general, valor_inf, valor_fobjetivo


f = "iris_set.dat"
f2 = "iris_set_const_10.const"

datos, restricciones, n, d = cargar_datos(f, f2)




v_sol, desv, valor_inf, valor_fobjetivo = AGG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)

print("EL vector solucion es: ")
print(v_sol)
print("Valor deviacion ", desv)
print("Valor infeasibility ", valor_inf)
print("Valor funcion objetivo ", valor_fobjetivo)
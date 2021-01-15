# enconding: utf-8

import numpy as np
import random
import math
import time
import pandas as pd
import os
from os import listdir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


semillas = np.array([1, 3, 2, 7, 5])

# Funcion que carga lo datos. Tanto el set de valores como la matriz de restricciones
# Se devuelve 4 parametros, la matriz de datos, la matriz de restricciones, y el numero de filas
# y columnas de nuestra matriz de datos

def cargar_datos(f_datos, f_restricciones):
    # Read data separated by commmas in a text file
    # with open("text.txt", "r") as filestream:

    #f = open(f_datos, "r")
    #f1 = open(f_restricciones, "r")
     
 
    string1 = "datos/" + f_datos 
    string2 = "restricciones/" + f_restricciones

    f = open(string1, "r")
    f1 = open(string2, "r")

   
   
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

    v_restricciones = []
    for i in range(len(matrix_const)):
        for j in range(i+1,len(matrix_const[0])):
            if matrix_const[i][j] == '-1':
                v_restricciones.append([i,j,-1])
            elif matrix_const[i][j] == '1':
                v_restricciones.append([i,j,1])

    return matrix_datos, v_restricciones, n, d








def generar_solucion_aleatoria(k,n):
    solucion = np.zeros(n)
    
    for i in range(n):
        solucion[i] = random.randint(1,k)

    return solucion







def reparacion(solucion, num_cluster,n_genes):
    posicion = random.randint(0,(n_genes-1))

    solucion[posicion] = num_cluster






def reparar(solucion,n):
  esta_vacio = True

  while esta_vacio:
    esta_vacio = False
    for i in range(k):
      if cluster_vacio(solucion, i+1) == 0:
        reparacion(solucion,i+1,n)
        esta_vacio = True
        
        

# Funcion que calcula la infeasibility.
# Si en la matriz de restricciones marca un 1 y dos instancias NO estan en el mismo cluster
# o si en la matriz de restricciones hay un -1 y estan en el mismo, sumamos +1 el valor de la
# infeasibility ya que sera una RESTRICCION INCUMPLIDA


def calcular_infeasibility(datos, valores_asignados, restricciones):
    infeasability = 0

    for i in range(len(restricciones)):
        if restricciones[i][2] == -1 and valores_asignados[restricciones[i][0]] == valores_asignados[restricciones[i][1]]:
            infeasability = infeasability + 1
        elif restricciones[i][2] == 1 and valores_asignados[restricciones[i][0]] != valores_asignados[restricciones[i][1]]:
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
            dist = 0

            if(distancia > maximo):
                maximo = distancia

    return maximo



# Funcion que calcula el numero de restricciones de la matriz de restricciones

def calcular_nrestricciones(restricciones, n):

    num_restricciones = len(restricciones)

    return num_restricciones



# Funcion que calcula el valor de lambda de la funcion objetivo 

def calcular_lambda(datos, restricciones, n):
    distancima_maxima = calcular_dmaxima(datos,n)

    num_restricciones = calcular_nrestricciones(restricciones, n)

    lamda = (int(distancima_maxima))/ num_restricciones

    return lamda


# Funcion que calcula la distancia

def distancia_media(datos, pos, centroides, vector_asignados):
    dist = 0
    elementos = 0
   
    for j in range (len(datos)):
        if vector_asignados[j] == pos + 1:
            dist = dist + np.linalg.norm(datos[j] - centroides[pos])
            elementos = elementos + 1
    dist = dist/elementos
    return dist

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

        if (cluster_vacio(valores_asignados, i + 1 ) != 0):
            desv = desv + distancia_media(datos, i, centroides, valores_asignados)
            cont = cont + 1

    desviacion_general = desv / cont

    return desviacion_general




# FUNCION OBJETIVO DE LA BUSQUEDA LOCAL    

def f_objetivo(datos, valores_asignados, restricciones, n, d, centroides, valor_lambda):
    return desviacion(datos, centroides, d, valores_asignados) + (calcular_infeasibility(datos, valores_asignados, restricciones) * valor_lambda)





def mutacion(solucion,n):
  #Realizaremos una mutacion por segmento
  solucion_mutada = np.copy(solucion)

  inicio_seg = random.randint(0,(n-1))
  tamanio = (int(0.1*n))
  fin_seg = ((inicio_seg + tamanio)%n)

  contador = inicio_seg

  while contador != fin_seg:
    valor = random.randint(1,k)
    solucion_mutada[contador] = valor

    contador = (contador+1)%n

  #for i in range(k):
    #if cluster_vacio(solucion_mutada,i+1) == 0:
      #reparacion(solucion_mutada,i+1,n)

  reparar(solucion_mutada,n)


  return solucion_mutada
  
 
    


"""

    ALGORITMO BUSQUEDA LOCAL

"""




def LocalSearch(datos, restricciones, n, d, v_solucion, centroides, valor_lambda):
  num_evaluaciones = 0
  hay_cambio = 1

  vecinos = []

  v_centroides = []
  for i in range(n*(k-1)):
      v_centroides.append(np.zeros((k,d)))
      

  #Vectores usados para realizar las graficas
  #valores_fobj = []
  #num_iterw = []

  coste_sol = f_objetivo(datos, v_solucion, restricciones, n, d, centroides, valor_lambda)

  while(hay_cambio == 1) and (num_evaluaciones < NUM_MAX_EVALUACIONES):
    hay_cambio = 0
    #print("llevo ", num_evaluaciones , " evaluaciones")
    vecinos = []
    for i in range(n):
        for j in range(k):
        
            if (float(j + 1)) != v_solucion[i]:

                vecinos.append(v_solucion.copy()) 
                valor_anterior = vecinos[-1][i]                 
                vecinos[-1][i] = j + 1

                if (cluster_vacio(vecinos[-1], valor_anterior ) == 0):
                    vecinos.pop()

    np.random.shuffle(vecinos)

    for v in range(len(vecinos)):
        recalcular_centroides(datos, vecinos[v], v_centroides[v], d)

        sol1 = f_objetivo(datos, vecinos[v], restricciones, n, d, v_centroides[v], valor_lambda)
        num_evaluaciones = num_evaluaciones + 1

        #sol2 = f_objetivo(datos, v_solucion, restricciones, n, d, centroides, valor_lambda)

        if sol1 < coste_sol :
            v_solucion = vecinos[v]
            centroides = v_centroides[v]
            coste_sol = sol1
            #valores_fobj.append(sol1)
            #num_iterw.append(contador)
            
            hay_cambio = 1
            break

        

  return v_solucion, coste_sol, centroides


"""

    ALGORITMOS PRACTICA 3
    
"""



def Enfriamiento_Simulado(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos):
  num_eval = 0
  num_exitos = 1

  la_solucion = generar_solucion_aleatoria(k,n)

  valor_lambda = calcular_lambda(datos, restricciones, n)
  centroides = np.zeros((k,d))
  centroides_vecinos = np.zeros((k,d))
  recalcular_centroides(datos, la_solucion, centroides, d)

  coste_solucion = f_objetivo(datos, la_solucion, restricciones, n, d, centroides, valor_lambda)
  num_eval = num_eval + 1

  mejor_solucion = np.copy(la_solucion)
  mejor_centroides = np.copy(centroides)
  coste_mejor = coste_solucion

  temperatura_inicial = (mu * coste_solucion) / -(np.log(phi))
  temperatura = np.copy(temperatura_inicial)
  valor_M = NUM_MAX_EVALUACIONES / max_vecinos
  beta = (temperatura_inicial - temperatura_final) / (valor_M*temperatura_inicial*temperatura_final)

  while num_exitos != 0 and num_eval <= NUM_MAX_EVALUACIONES:
    num_exitos = 0
    num_vecinos = 0

    while num_vecinos < max_vecinos and num_exitos < max_exitos:

      # REALIZAMOS LA EXPLORACION DEL VECINADIRO
      vecino = np.copy(la_solucion)
      posicion = random.randint(0,(n-1))

      valor = vecino[posicion]
      nuevo_valor = random.randint(1,k)

      while nuevo_valor == valor:
         nuevo_valor = random.randint(1,k)

      vecino[posicion] = nuevo_valor

      reparar(vecino,n)

      num_vecinos = num_vecinos + 1

      recalcular_centroides(datos, vecino, centroides_vecinos, d)
      coste_vecino = f_objetivo(datos, vecino, restricciones, n, d, centroides_vecinos, valor_lambda)
      num_eval = num_eval + 1
      diferencia = coste_vecino - coste_solucion

      aleatorio = random.random()

      if diferencia < 0 or aleatorio <= math.exp(-diferencia / temperatura):
        la_solucion = np.copy(vecino)
        recalcular_centroides(datos, la_solucion, centroides, d)
        coste_solucion = f_objetivo(datos, la_solucion, restricciones, n, d, centroides, valor_lambda)
        num_eval = num_eval + 1


        num_exitos = num_exitos + 1

        if coste_solucion < coste_mejor:
          mejor_solucion = np.copy(la_solucion)
          mejor_centroides = np.copy(centroides)
          coste_mejor = coste_solucion

    #Esquema de enfriamiento --> Esquema de Cauchy
    temperatura = temperatura / (1 + beta * temperatura)

  desviacion_general = desviacion(datos,mejor_centroides, d, mejor_solucion)
  valor_inf = calcular_infeasibility(datos, mejor_solucion, restricciones)
  valor_fobjetivo = f_objetivo(datos, mejor_solucion, restricciones, n, d, mejor_centroides, valor_lambda)

  if ES_act == False:
    print("La desviacion general es: ", desviacion_general)
    print("valor de infeasibility ", valor_inf)
    print("valor f_obj: ", valor_fobjetivo)

  return mejor_solucion, valor_fobjetivo, mejor_centroides



"""
def Enfriamiento_Simulado_2(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos,alpha):
  num_eval = 0
  num_exitos = 1

  la_solucion = generar_solucion_aleatoria(k,n)

  valor_lambda = calcular_lambda(datos, restricciones, n)
  centroides = np.zeros((k,d))
  centroides_vecinos = np.zeros((k,d))
  recalcular_centroides(datos, la_solucion, centroides, d)

  coste_solucion = f_objetivo(datos, la_solucion, restricciones, n, d, centroides, valor_lambda)
  num_eval = num_eval + 1

  mejor_solucion = np.copy(la_solucion)
  mejor_centroides = np.copy(centroides)
  coste_mejor = coste_solucion

  temperatura_inicial = (mu * coste_solucion) / -(np.log(phi))
  temperatura = np.copy(temperatura_inicial)
  valor_M = NUM_MAX_EVALUACIONES / max_vecinos
  beta = (temperatura_inicial - temperatura_final) / (valor_M*temperatura_inicial*temperatura_final)

  while num_exitos != 0 and num_eval <= NUM_MAX_EVALUACIONES:
    num_exitos = 0
    num_vecinos = 0

    while num_vecinos < max_vecinos and num_exitos < max_exitos:

      # REALIZAMOS LA EXPLORACION DEL VECINADIRO
      vecino = np.copy(la_solucion)
      posicion = random.randint(0,(n-1))

      valor = vecino[posicion]
      nuevo_valor = random.randint(1,k)

      while nuevo_valor == valor:
         nuevo_valor = random.randint(1,k)

      vecino[posicion] = nuevo_valor

      #for i in range(k):
        #if cluster_vacio(vecino,i+1) == 0:
          #reparacion(vecino,i+1,n)
          #break
      reparar(vecino,n)

      num_vecinos = num_vecinos + 1

      recalcular_centroides(datos, vecino, centroides_vecinos, d)
      coste_vecino = f_objetivo(datos, vecino, restricciones, n, d, centroides_vecinos, valor_lambda)
      num_eval = num_eval + 1
      diferencia = coste_vecino - coste_solucion

      aleatorio = random.random()

      if diferencia < 0 or aleatorio <= math.exp(-diferencia / temperatura):
        la_solucion = np.copy(vecino)
        recalcular_centroides(datos, la_solucion, centroides, d)
        coste_solucion = f_objetivo(datos, la_solucion, restricciones, n, d, centroides, valor_lambda)
        num_eval = num_eval + 1


        num_exitos = num_exitos + 1

        if coste_solucion < coste_mejor:
          mejor_solucion = np.copy(la_solucion)
          mejor_centroides = np.copy(centroides)
          coste_mejor = coste_solucion

    #Esquema de enfriamiento --> Esquema de Cauchy
    #temperatura = temperatura / (1 + beta * temperatura)
    temperatura = temperatura * alpha

  desviacion_general = desviacion(datos,mejor_centroides, d, mejor_solucion)
  valor_inf = calcular_infeasibility(datos, mejor_solucion, restricciones)
  valor_fobjetivo = f_objetivo(datos, mejor_solucion, restricciones, n, d, mejor_centroides, valor_lambda)

  if ES_act == False:
    print("La desviacion general es: ", desviacion_general)
    print("valor de infeasibility ", valor_inf)
    print("valor f_obj: ", valor_fobjetivo)
    #return mejor_solucion

  return mejor_solucion, valor_fobjetivo, mejor_centroides

"""



          
   


def Busqueda_Multiarranque(datos, restricciones, n, d):  
  mejor_solucion = np.zeros(n)
  coste_mejor = float("INF")

  valor_lambda = calcular_lambda(datos, restricciones, n)

  for i in range(MAX_ITER):
    solucion = generar_solucion_aleatoria(k,n)
    centroides = np.zeros((k,d))
    recalcular_centroides(datos, solucion, centroides, d)

    #Optimizar cada una de las evaluaciones con la BL
    solucion, coste_solucion, centroides = LocalSearch(datos, restricciones, n, d, solucion, centroides, valor_lambda)

    if coste_solucion < coste_mejor:
      coste_mejor = coste_solucion
      mejor_solucion = np.copy(solucion)
      mejor_centroides = np.copy(centroides)


  desviacion_general = desviacion(datos,mejor_centroides, d, mejor_solucion)
  print("La desviacion general es: ", desviacion_general)
  valor_inf = calcular_infeasibility(datos, mejor_solucion, restricciones)
  print("valor de infeasibility ", valor_inf)
  valor_fobjetivo = coste_mejor
  print("valor f_obj: ", valor_fobjetivo)







def ILS(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos, ES_act):
  iteraciones = 9
  valor_lambda = calcular_lambda(datos, restricciones, n)
  centroides_mutados = np.zeros((k,d))

  solucion_inicial = generar_solucion_aleatoria(k,n)
  centroides = np.zeros((k,d))
  recalcular_centroides(datos, solucion_inicial, centroides, d)
  coste_inicial = f_objetivo(datos, solucion_inicial, restricciones, n, d, centroides, valor_lambda)

  solucion = np.zeros(n)

  if ES_act == False:
    #print("ALGORITMO ILS")
    solucion, coste_solucion, centroides = LocalSearch(datos, restricciones, n, d, solucion_inicial, centroides,valor_lambda)
  else:
    #print("ALGORITMO ILS-ES")
    solucion, coste_solucion, centroides = Enfriamiento_Simulado(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos)

  if coste_inicial < coste_solucion:
    mejor_solucion = np.copy(solucion_inicial)
    mejor_coste = coste_inicial
  else:
    mejor_solucion = np.copy(solucion)
    mejor_coste = coste_solucion
    mejor_centroides = centroides

  for i in range(iteraciones):
    solucion_mutada = mutacion(mejor_solucion,n)
    recalcular_centroides(datos, solucion_mutada, mejor_centroides, d)
    centroides_mutados = np.copy(mejor_centroides)
  
    if ES_act == False:
      solucion, coste_solucion, centroides = LocalSearch(datos, restricciones, n, d, solucion_mutada, centroides_mutados,valor_lambda)
    else:
      solucion, coste_solucion, centroides = Enfriamiento_Simulado(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos)

    if coste_solucion < mejor_coste:
      mejor_solucion = np.copy(solucion)
      mejor_coste = coste_solucion
      #print("MEJOR COSTE ", mejor_coste)
      mejor_centroides = centroides


  desviacion_general = desviacion(datos,mejor_centroides, d, mejor_solucion)
  print("La desviacion general es: ", desviacion_general)
  valor_inf = calcular_infeasibility(datos, mejor_solucion, restricciones)
  print("valor de infeasibility ", valor_inf)
  valor_fobjetivo = mejor_coste
  print("valor f_obj: ", valor_fobjetivo)






print("ELECCION DEL CONJUNTO DE DATOS")
print(" Introduzca: 1 -- CONJUNTO IRIS , 2 -- CONJUNTO ECOLI, 3 -- CONJUNTO RAND, 4 -- CONJUNTO NEWTHYROID")
numero = input("Introduce un numero:")

if numero == "1":
    f = "iris_set.dat"
elif numero == "2":
    f = "ecoli_set.dat"
elif numero == "3":
    f = "rand_set.dat"
elif numero == "4":
    f = "newthyroid_set.dat"
else:
    print("ERROR - Opción no valida")

txt_datos = f.split("_")

if txt_datos[0] == 'iris' or txt_datos[0] == 'rand' or txt_datos[0] == 'newthyroid':
    k = 3
elif txt_datos[0] == 'ecoli':
    k = 8


for f2 in listdir("restricciones"):
    txt_restricciones = f2.split("_")

    if txt_datos[0] == txt_restricciones[0]:    

        
        for c in range(5):
            print("CONJUNTO DE DATOS " , f , " PARA CONJUNTO DE RESTRICCIONES ", f2)
            
            print("Usamos la semilla " , semillas[c])
            random.seed(semillas[c])

            datos, restricciones, n, d = cargar_datos(f, f2)
            
           
            ES_act = False 
            #ALGORITMO ENFRIAMIENTO SIMULADO 
         
            NUM_MAX_EVALUACIONES = 100000
            mu = 0.3
            phi = 0.3
            temperatura_final = 10**(-3)

            max_vecinos = 10 * n
            max_exitos = 0.1 * max_vecinos
         
            print("Ejecutamos el algoritmo ES")
            ini = time.time()
            Enfriamiento_Simulado(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos)
            fin = time.time() - ini
            print("Tiempo ES: ", fin)
           
           
          
            """
            print(" BONUS ")
            print("probaremos con max_vecinos = 5* n")
         
            max_vecinos = 5 * n 
            ini = time.time()
            Enfriamiento_Simulado(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos)
            fin = time.time() - ini
            print("Tiempo ES: ", fin)
         
            print()
            print()
         
            print("Tambien probaremos usando un esquima de enfriamiendo de Tk+1 = T*alpha, donde alpha = 0.9")
            alpha = 0.9
            ini = time.time()
            Enfriamiento_Simulado_2(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos, alpha)
            fin = time.time() - ini
            print("Tiempo ES: ", fin)
         
            print("-------------------------------------------------")
            """
         
         
         
            print()
            print()
         
            #BUSQUEDA MULTIARRANQUE BASICA
         
            NUM_MAX_EVALUACIONES = 10000
            MAX_ITER = 10
         
            print("Ejecutamos el algoritmo BMB")
            ini = time.time()
            Busqueda_Multiarranque(datos, restricciones, n, d)
            fin = time.time() - ini
            print("Tiempo BMB ", fin)
         
            print()
            print()
         
            #BUSQUEDA LOCAL REITERADA
         
            print("Ejecutamos el algoritmo ILS")
         
         
            NUM_MAX_EVALUACIONES = 10000
            ini = time.time()
            ILS(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos, ES_act)
            fin = time.time() - ini
            print("Tiempo ILS ", fin)
         
            print()
            print()
         
            #ALGORITMO ILS-ES
         
            print("Ejecutamos el algoritmo ILS-ES")
         
            ES_act = True
         
            ini = time.time()
            ILS(datos, restricciones, n, d, mu, phi, temperatura_final, max_vecinos, max_exitos, ES_act)
            fin = time.time() - ini
            print("Tiempo ILS-ES " , fin )
            
          

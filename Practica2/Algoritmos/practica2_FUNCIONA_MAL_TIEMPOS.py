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


NUM_MAX_EVALUACIONES = 100000
semillas = np.array([1, 3, 2, 7, 5])


class Cromosoma():
    def __init__(self,n,d):
        self.valores_asignados = np.zeros(n)
        self.centroides = np.zeros((k,d))
        self.vf_objetivo = 0.0

    def __lt__(self, cromosoma):
        return self.vf_objetivo < cromosoma.vf_objetivo


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

    """
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
    """
def evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones):
    for i in range(long_poblacion):
        recalcular_centroides(datos, poblacion[i].valores_asignados, poblacion[i].centroides, d)

        poblacion[i].vf_objetivo = f_objetivo(datos, poblacion[i].valores_asignados, restricciones, n, d, poblacion[i].centroides, valor_lambda)
        num_evaluaciones = num_evaluaciones + 1
    
    return num_evaluaciones

def calcula_mejor_peor(poblacion, long_poblacion):
    mejor_valor = float("inf")
    peor_valor = float("-inf")

    for i in range(long_poblacion):
        if(poblacion[i].vf_objetivo < mejor_valor):
            mejor_valor = poblacion[i].vf_objetivo
            mejor_cromosoma = poblacion[i]
            pos_mejor = i

        if(poblacion[i].vf_objetivo > peor_valor):
            peor_valor = poblacion[i].vf_objetivo
            peor_cromosoma = poblacion[i]    
            pos_peor = i

    return mejor_cromosoma, peor_cromosoma,pos_mejor, pos_peor


def seleccionar_poblacion(long_poblacion,poblacion):
    #Seleccion por torneo: se eligen aleatoriamente dos inidividuos de la poblacion 
    #y quedarse con el mejor de ellos
    #GENERACIONAL: se aplicaran tantos torneos como individuos existan en la poblacion
    #incluyendo los individuos ganadores

    num_torneos = 0
    nueva_poblacion = []
    #nueva_poblacion = np.zeros(long_poblacion,n)

    if elitista == True:
        tope = long_poblacion
    else:
        tope = 2
    

    while(num_torneos < tope):
        aleatorio1 = random.randint(0,long_poblacion-1)
        aleatorio2 = random.randint(0,long_poblacion-1)

        if poblacion[aleatorio1].vf_objetivo < poblacion[aleatorio2].vf_objetivo:
            mejor_crom = poblacion[aleatorio1]
            nueva_poblacion.append(mejor_crom)
        else :
            mejor_crom = poblacion[aleatorio2]
            nueva_poblacion.append(mejor_crom)

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

    #p_intermedia = []
    p_intermedia = np.copy(poblacion)
    
    if elitista == True:
        while contador < num_esperado_cruces:

            #Generamos dos hijos

            if coste_uniforme == True:
                hijo = operador_uniforme(poblacion[individuo], poblacion[individuo+1], n_genes,d)
                otro_hijo = operador_uniforme(poblacion[individuo], poblacion[individuo+1], n_genes,d)
            else:
                hijo = segmento_fijo(poblacion[individuo], poblacion[individuo+1], n_genes,d)
                otro_hijo = segmento_fijo(poblacion[individuo], poblacion[individuo+1], n_genes,d)

            #reparar = True
            #while reparar:
                #reparar = False

            for i in range(k):
                if cluster_vacio(hijo.valores_asignados,i+1) == 0:
                    #reparar = True
                    reparacion(hijo,i+1,n_genes)
                    break
            
            #p_intermedia.append(hijo)
            p_intermedia[individuo] = hijo

            #reparar = True
            #while reparar:
                #reparar = False

            for i in range(k):
                if cluster_vacio(otro_hijo.valores_asignados,i+1) == 0:
                    #reparar = True
                    reparacion(otro_hijo,i+1,n_genes)
                    break
            
            #p_intermedia.append(otro_hijo)
            p_intermedia[individuo + 1] = otro_hijo

            individuo = individuo+2
            contador = contador+1
    else:
        if coste_uniforme == True:
            hijo = operador_uniforme(poblacion[0], poblacion[1], n_genes,d)
            otro_hijo = operador_uniforme(poblacion[0], poblacion[1], n_genes,d)
        else:
            hijo = segmento_fijo(poblacion[0], poblacion[1], n_genes,d)
            otro_hijo = segmento_fijo(poblacion[0], poblacion[1], n_genes,d)


        for i in range(k):
            if cluster_vacio(hijo.valores_asignados,i+1) == 0:
                #reparar = True
                reparacion(hijo,i+1,n_genes)
                break
        
        #p_intermedia.append(hijo)
        p_intermedia[0] = hijo

        for i in range(k):
            if cluster_vacio(otro_hijo.valores_asignados,i+1) == 0:
                #reparar = True
                reparacion(otro_hijo,i+1,n_genes)
                break

        #p_intermedia.append(otro_hijo)
        p_intermedia[1] = otro_hijo

    return p_intermedia


def mutacion(num_esperado_mutacion, poblacion ,n_genes,prob_muta, long_poblacion):
    #Calculo el cromosoma a realizar la primera mutacion
    mu_next = math.ceil(math.log(random.random()) / math.log(1.0 - prob_muta))
    contador = 0

    if elitista == True:
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

    else:
        aleatorio_crom1 = random.random()
        aleatorio_crom2 = random.random()

        if aleatorio_crom1 < num_esperado_mutacion or aleatorio_crom2 < num_esperado_mutacion:
            if aleatorio_crom1 < num_esperado_mutacion:
                crom = 0
            elif aleatorio_crom2 < num_esperado_mutacion:
                crom = 1

            gen_muta = random.randint(0,(n_genes - 1))

            cluster_individuo = poblacion[crom].valores_asignados[gen_muta]
            nuevo_valor = random.randint(1,k)

            poblacion[crom].valores_asignados[gen_muta] = nuevo_valor

            while nuevo_valor == cluster_individuo:
                cluster_individuo = poblacion[crom].valores_asignados[gen_muta]
                nuevo_valor = random.randint(1,k)
                poblacion[crom].valores_asignados[gen_muta] = nuevo_valor

    return poblacion


def encontrar_cromosoma(poblacion, cromosoma):
    for i in range(len(poblacion)):
        if poblacion[i].vf_objetivo == cromosoma.vf_objetivo:
            if (poblacion[i].valores_asignados == cromosoma.valores_asignados).all():
                return True

    return False

def segundo_peor(poblacion, peor_cromosoma):
    peor_valor = float("-inf")
    segundo_peor = poblacion[len(poblacion)-1]
    pos_peor = len(poblacion)-1

    for i in range(len(poblacion)):
        if poblacion[i].vf_objetivo >= peor_valor and poblacion[i].vf_objetivo != peor_cromosoma.vf_objetivo:
            peor_valor = poblacion[i].vf_objetivo
            segundo_peor = poblacion[i]
            pos_peor = i

            
    return segundo_peor, pos_peor



def reemplazar_poblacion(poblacion_actual, poblacion_hijos, mejor_cromosoma, peor_cromosoma, pos_peor, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones):
    
    if elitista == True:
        # La poblacion de hijos sustituye automaticamente a la actual
        # Si la mejor sol. de la generacion anterior no sobrevive, sustituye 
        # directamente la peor de la poblacion

        poblacion_actual = np.copy(poblacion_hijos)

        #nuevo_mejor, nuevo_peor = evaluar_poblacion(poblacion_actual, long_poblacion, datos, restricciones, n, d, valor_lambda)

        if encontrar_cromosoma(poblacion_actual, mejor_cromosoma) == False:
            #nuevo_mejor, nuevo_peor, pos_mejor, pos_peor, num_evaluaciones = evaluar_poblacion(poblacion_actual, long_poblacion, datos, restricciones, n, d, valor_lambda,num_evaluaciones)
            num_evaluaciones = evaluar_poblacion(poblacion_actual, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
            nuevo_mejor, nuevo_peor, pos_mejor, pos_peor = calcula_mejor_peor(poblacion_actual,long_poblacion)
            
            poblacion_actual[pos_peor].valores_asignados = np.copy(mejor_cromosoma.valores_asignados)
            poblacion_actual[pos_peor].vf_objetivo = np.copy(mejor_cromosoma.valores_asignados)
            poblacion_actual[pos_peor].centroides = np.copy(mejor_cromosoma.centroides)
    
    else:
        #print("V.objetivo del MEJOR cromosoma ", mejor_cromosoma.vf_objetivo)
        #print("V.objetivo del PEOR cromosoma ", peor_cromosoma.vf_objetivo,  " en la posicion ", pos_peor)

        peor_cromosoma2, pos_peor2 = segundo_peor(poblacion_actual, peor_cromosoma)

        num_evaluaciones = evaluar_poblacion(poblacion_hijos, len(poblacion_hijos), datos, restricciones, n, d, valor_lambda, num_evaluaciones)
        mejor_hijo, mejor_hijo2, pos_mejor_hijo, pos_mejor_hijo2 = calcula_mejor_peor(poblacion_hijos,len(poblacion_hijos))

        #print("V.objetivo SEGUNDO VALOR ", peor_cromosoma2.vf_objetivo , " en la posicion ", pos_peor2 )
        #mejor_hijo, mejor_hijo2, pos_mejor_hijo, pos_mejor_hijo2, num_evaluaciones = evaluar_poblacion(poblacion_hijos, len(poblacion_hijos), datos, restricciones, n, d, valor_lambda,num_evaluaciones)

        #if poblacion_hijos[0].vf_objetivo < poblacion_hijos[1].vf_objetivo:
            #el_mejor = 0
            #segundo_mejor = 1
        #else:
            #el_mejor = 1
            #segundo_mejor = 0

        #print("DE LA POBLACION DE HIJOS")
        #print("V.objetivo del MEJOR cromosoma ", poblacion_hijos[pos_mejor_hijo].vf_objetivo , " en posicion ", pos_mejor_hijo)
        #print("V.objetivo SEGUNDO VALOR ", poblacion_hijos[pos_mejor_hijo2].vf_objetivo, " en posicion ", pos_mejor_hijo2)

        #Si el mejor de los hijos es mejor que el peor de los padres
        if poblacion_hijos[pos_mejor_hijo].vf_objetivo < peor_cromosoma.vf_objetivo:
            #print("el mejor hijo es mejor que el peor padre ")
            #print("valor MEJOR hijo ", poblacion_hijos[pos_mejor_hijo].vf_objetivo)
            #print("valor PEOR padre" , peor_cromosoma.vf_objetivo)
            #print()

            poblacion_actual[pos_peor].valores_asignados = np.copy(poblacion_hijos[pos_mejor_hijo].valores_asignados)
            poblacion_actual[pos_peor].vf_objetivo = np.copy(poblacion_hijos[pos_mejor_hijo].vf_objetivo)
            poblacion_actual[pos_peor].centroides = np.copy(poblacion_hijos[pos_mejor_hijo].centroides)

            #Si el segundo mejor es mejor que el segundo peor
            if poblacion_hijos[pos_mejor_hijo2].vf_objetivo < peor_cromosoma2.vf_objetivo:
                #print("el  segundo mejor hijo es mejor que el segundo peor padre ")
                #print("valor SEGUNDO MEJOR hijo ", poblacion_hijos[pos_mejor_hijo2].vf_objetivo)
                #print("valor SEGUNDO PEOR padre" , peor_cromosoma2.vf_objetivo)
                #print()
                #print()
                poblacion_actual[pos_peor2].valores_asignados = np.copy(poblacion_hijos[pos_mejor_hijo2].valores_asignados)
                poblacion_actual[pos_peor2].vf_objetivo = np.copy(poblacion_hijos[pos_mejor_hijo2].vf_objetivo)
                poblacion_actual[pos_peor2].centroides = np.copy(poblacion_hijos[pos_mejor_hijo2].centroides)   
    
    return poblacion_actual, num_evaluaciones


"""

    BUSQUEDA LOCAL SUAVE - ALGORITMO MEMETICO

"""

def BL_suave(cromosoma, k, num_fallos_max, n_genes, d, datos, restricciones, valor_lambda, num_evaluaciones):
    rsi = np.arange(n_genes)
    np.random.shuffle(rsi)

    fallos = 0
    mejora = True
    i = 0

    vf_antiguo = cromosoma.vf_objetivo

    while (mejora or fallos < num_fallos_max) and i < n_genes:
        mejora = False

        cluster_act = cromosoma.valores_asignados[rsi[i]]
        for j in range(k):
            if ((float(j+1)) != cluster_act):
                cromosoma.valores_asignados[rsi[i]] = j + 1
                
                # Si al cambiar el cluster del gen, el cluster se queda vacio
                #no realizaremos el cambio

                if (cluster_vacio(cromosoma.valores_asignados, cluster_act)) == 0:
                    cromosoma.valores_asignados[rsi[i]] = cluster_act
                else:

                    recalcular_centroides(datos, cromosoma.valores_asignados, cromosoma.centroides, d)

                    cromosoma.vf_objetivo = f_objetivo(datos, cromosoma.valores_asignados, restricciones, n, d, cromosoma.centroides, valor_lambda)
                    num_evaluaciones = num_evaluaciones + 1


                    if cromosoma.vf_objetivo < vf_antiguo:
                        mejora = True

        if mejora == False:
            fallos = fallos + 1

        i = i + 1               

    return cromosoma



"""

    ALGORITMOS GENÉTICOS Y MEMÉTICOS

"""

def AG(datos, restricciones, n, d, long_poblacion, prob_cruce, prob_muta):
    num_evaluaciones = 0
    num_esperado_cruces = int(prob_cruce * (long_poblacion/2))
    if elitista == True:
        num_esperado_mutacion = int(prob_muta * (long_poblacion*n))
    else:
        # Para cada cromosoma, generare un numero aleatorio [0,1]
        # Si el numero < "num_esperado_mutacion " -> mutara
        num_esperado_mutacion = prob_muta * n

    poblacion = generar_poblacion_aleatoria(long_poblacion, k, n, d)

    valor_lambda = calcular_lambda(datos, restricciones, n)

    #Evaluamos los individuos de la población inicial
    num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
    mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor = calcula_mejor_peor(poblacion,long_poblacion)


    cuenta_selec = 0
    cuenta_cruce = 0
    cuenta_muta = 0
    cuenta_reemp = 0
    cuenta_ev = 0
    cuenta_mejp = 0

    while(num_evaluaciones <= NUM_MAX_EVALUACIONES):
        print(num_evaluaciones)
        #Seleccionar P' desde P(t-1)

        inis = time.time()
        nueva_poblacion = seleccionar_poblacion(long_poblacion,poblacion)
        fins = time.time() - inis

        cuenta_selec = cuenta_selec + fins

        inicr = time.time()
        p_cruzada = cruce(num_esperado_cruces, nueva_poblacion,n,d)
        fincr = time.time() - inicr

        cuenta_cruce = cuenta_cruce + fincr

        inim = time.time()
        p_hijos = mutacion(num_esperado_mutacion, p_cruzada,n,prob_muta,long_poblacion)
        finm = time.time() - inim

        cuenta_muta = cuenta_muta + finm

        inir = time.time()
        poblacion, num_evaluaciones = reemplazar_poblacion(poblacion, p_hijos, mejor_cromosoma, peor_cromosoma, pos_peor, long_poblacion,datos,restricciones,n,d,valor_lambda, num_evaluaciones)  
        finr = time.time() - inir

        cuenta_reemp = cuenta_reemp + finr

        if elitista == True:
            iniev = time.time()
            num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
            finev = time.time() - iniev
            cuenta_ev = cuenta_ev + finev

        inimp = time.time()
        mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor = calcula_mejor_peor(poblacion,long_poblacion)
        finmp = time.time() - inimp

        cuenta_mejp = cuenta_mejp + finmp


    print("Tiempo total de seleccion ", cuenta_selec)
    print("Tiempo total de cruce ", cuenta_cruce)
    print("Tiempo total de mutacion ", cuenta_muta)
    print("Tiempo total de reemplazo ", cuenta_reemp)
    print("Tiempo total de evaluacion ", cuenta_ev)
    print("Tiempo total de mejor peor cromosoma ", cuenta_mejp)
    

    desviacion_general = desviacion(datos, mejor_cromosoma.centroides, d, mejor_cromosoma.valores_asignados)
    valor_inf = calcular_infeasibility(datos, mejor_cromosoma.valores_asignados, restricciones)
    valor_fobjetivo = f_objetivo(datos, mejor_cromosoma.valores_asignados, restricciones, n, d, mejor_cromosoma.centroides, valor_lambda)

    return mejor_cromosoma.valores_asignados, desviacion_general, valor_inf, valor_fobjetivo





def AM(generaciones, tasa, tipo_mej, datos, restricciones, n, d, long_poblacion, prob_cruce, prob_muta):
    num_evaluaciones = 0
    num_generaciones = 0

    num_esperado_cruces = int(prob_cruce * (long_poblacion/2))
    num_esperado_mutacion = int(prob_muta * (long_poblacion*n))
    valor_lambda = calcular_lambda(datos, restricciones, n)
    num_fallos_max = 0.1*n

    poblacion = generar_poblacion_aleatoria(long_poblacion, k, n, d)
    #Evaluamos los individuos de la población inicial
    #mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor , num_evaluaciones= evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
    num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
    mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor = calcula_mejor_peor(poblacion,long_poblacion)

    while(num_evaluaciones <= NUM_MAX_EVALUACIONES):
        #print(num_evaluaciones)
        #Seleccionar P' desde P(t-1)
        nueva_poblacion = seleccionar_poblacion(long_poblacion,poblacion)
        p_cruzada = cruce(num_esperado_cruces, nueva_poblacion,n,d)
        p_hijos = mutacion(num_esperado_mutacion, p_cruzada,n,prob_muta,long_poblacion)


        #Cada 10 generaciones -> aplicar una busqueda local
        if (num_generaciones%generaciones) == 0:
            if tipo_mej == True:
                #mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor, num_evaluaciones = evaluar_poblacion(p_hijos, len(p_hijos), datos, restricciones, n, d, valor_lambda, num_evaluaciones)
                num_evaluaciones = evaluar_poblacion(p_hijos, len(p_hijos), datos, restricciones, n, d, valor_lambda, num_evaluaciones)
                #mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor = calcula_mejor_peor(poblacion,long_poblacion)
                lista = sorted(p_hijos)

                for i in range(np.int(tasa*len(lista))):
                    p_hijos[i] = BL_suave(p_hijos[i], k, num_fallos_max, n, d, datos, restricciones, valor_lambda, num_evaluaciones)

            else:
                for i in range(len(p_hijos)):
                    valor = random.random()
                    if valor < tasa:
                        p_hijos[i] = BL_suave(p_hijos[i], k, num_fallos_max, n, d, datos, restricciones, valor_lambda, num_evaluaciones)


        poblacion, num_evaluaciones = reemplazar_poblacion(poblacion, p_hijos, mejor_cromosoma, peor_cromosoma, pos_peor, long_poblacion,datos,restricciones,n,d,valor_lambda, num_evaluaciones)  
        #mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor, num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
        num_evaluaciones = evaluar_poblacion(poblacion, long_poblacion, datos, restricciones, n, d, valor_lambda, num_evaluaciones)
        mejor_cromosoma, peor_cromosoma, pos_mejor, pos_peor = calcula_mejor_peor(poblacion,long_poblacion)

        num_generaciones = num_generaciones + 1

    desviacion_general = desviacion(datos, mejor_cromosoma.centroides, d, mejor_cromosoma.valores_asignados)
    valor_inf = calcular_infeasibility(datos, mejor_cromosoma.valores_asignados, restricciones)
    valor_fobjetivo = f_objetivo(datos, mejor_cromosoma.valores_asignados, restricciones, n, d, mejor_cromosoma.centroides, valor_lambda)

    return mejor_cromosoma.valores_asignados, desviacion_general, valor_inf, valor_fobjetivo
        


"""

for f in listdir("datos"):
    txt_datos = f.split("_")

    if txt_datos[0] == 'iris' or txt_datos[0] == 'rand' or txt_datos[0] == 'newthyroid':
        k = 3
    elif txt_datos[0] == 'ecoli':
        k = 8
        
    for f2 in listdir("restricciones"):
        txt_restricciones = f2.split("_")

        if txt_datos[0] == txt_restricciones[0]:
            matrix_AGG_UN = []
            matrix_AGG_SF = []
            #matrix_AGE_UN = []
            #matrix_AGE_SF = []
            matrix_AM_101 = []
            matrix_AM_1001 = []
            matrix_AM_1001M = []

            vector_AGG_UN = []
            vector_AGG_SF = []
            #vector_AGE_UN = []
            #vector_AGE_SF = []
            vector_AM_101 = []
            vector_AM_1001 = []
            vector_AM_1001M = []
            
            for c in range(5):
                print("CONJUNTO DE DATOS " , f , " PARA CONJUNTO DE RESTRICCIONES ", f2)
                
                print("Usamos la semilla " , semillas[c])
                random.seed(semillas[c])

                datos, restricciones, n, d = cargar_datos(f, f2)

                long_poblacion = 50
                prob_muta = 0.001

                elitista = True
                prob_cruce = 0.7

                print("INICIO DEL ALGORITMO GENETICO GENERACIONAL - CRUCE UNIFORME")
                coste_uniforme = True

                start_AGG_UN = time.time()
                v_sol_AGG_UN, desv_AGG_UN, valor_inf_AGG_UN, valor_fobjetivo_AGG_UN = AG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AGG_UN = time.time() - start_AGG_UN

                print("EL vector solucion es:")
                print(v_sol_AGG_UN)
                print("TIEMPO ", start_fin_AGG_UN)

                vector_AGG_UN = (desv_AGG_UN, valor_inf_AGG_UN, valor_fobjetivo_AGG_UN, start_fin_AGG_UN)


                print("INICIO DEL ALGORITMO GENETICO GENERACIONAL - CRUCE SEGMENTO FIJO")
                coste_uniforme = False

                start_AGG_SF = time.time()
                v_sol_AGG_SF, desv_AGG_SF, valor_inf_AGG_SF, valor_fobjetivo_AGG_SF = AG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AGG_SF = time.time() - start_AGG_SF

                print("EL vector solucion es:")
                print(v_sol_AGG_SF)
                print("TIEMPO ", start_fin_AGG_SF)

                vector_AGG_SF = (desv_AGG_SF, valor_inf_AGG_SF, valor_fobjetivo_AGG_SF, start_fin_AGG_SF)


                
                elitista = False
                prob_cruce = 1

                print("INICIO DEL ALGORITMO GENETICO ESTACIONARIO - CRUCE UNIFORME")
                coste_uniforme = True

                start_AGE_UN = time.time()
                v_sol_AGE_UN, desv_AGE_UN, valor_inf_AGE_UN, valor_fobjetivo_AGE_UN = AG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AGE_UN = time.time() - start_AGE_UN

                print("EL vector solucion es:")
                print(v_sol_AGE_UN)
                print("TIEMPO ", start_fin_AGE_UN)

                vector_AGE_UN = (desv_AGE_UN, valor_inf_AGE_UN, valor_fobjetivo_AGE_UN, start_fin_AGE_UN)

                print("INICIO DEL ALGORITMO GENETICO ESTACIONARIO - CRUCE SEGMENTO FIJO")
                coste_uniforme = False

                start_AGE_SF = time.time()
                v_sol_AGE_SF, desv_AGE_SF, valor_inf_AGE_SF, valor_fobjetivo_AGE_SF = AG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AGE_SF = time.time() - start_AGE_SF

                print("EL vector solucion es:")
                print(v_sol_AGE_SF)
                print("TIEMPO ", start_fin_AGE_SF)

                vector_AGE_SF = (desv_AGE_SF, valor_inf_AGE_SF, valor_fobjetivo_AGE_SF, start_fin_AGE_SF)
                

                long_poblacion = 10
                prob_muta = 0.001
                prob_cruce = 0.7

                elitista = True
                coste_uniforme = False

                generaciones = 10

                print("INICIO DEL ALGORITMO MEMETICO 10-1.0")               
                tasa = 1
                tipo_mej = False

                start_AM_101 = time.time()
                v_sol_AM_101, desv_AM_101, valor_inf_AM_101, valor_fobjetivo_AM_101 = AM(generaciones, tasa, tipo_mej, datos, restricciones,n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AM_101 = time.time() - start_AM_101

                print("EL vector solucion es:")
                print(v_sol_AM_101)
                print("TIEMPO ", start_fin_AM_101)

                vector_AM_101 = (desv_AM_101, valor_inf_AM_101, valor_fobjetivo_AM_101, start_fin_AM_101)

                print("INICIO DEL ALGORITMO MEMETICO 10-0.1")
                tasa = 0.1
                tipo_mej = False

                start_AM_1001 = time.time()
                v_sol_AM_1001, desv_AM_1001, valor_inf_AM_1001, valor_fobjetivo_AM_1001 = AM(generaciones, tasa, tipo_mej, datos, restricciones,n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AM_1001 = time.time() - start_AM_1001

                print("EL vector solucion es:")
                print(v_sol_AM_1001)
                print("TIEMPO ", start_fin_AM_1001)

                vector_AM_1001 = (desv_AM_1001, valor_inf_AM_1001, valor_fobjetivo_AM_1001, start_fin_AM_1001)

                print("INICIO DEL ALGORITMO MEMETICO 10-0.1MEJ")
                tasa = 0.1
                tipo_mej = True

                start_AM_1001M = time.time()
                v_sol_AM_1001M, desv_AM_1001M, valor_inf_AM_1001M, valor_fobjetivo_AM_1001M = AM(generaciones, tasa, tipo_mej, datos, restricciones,n,d,long_poblacion,prob_cruce,prob_muta)
                start_fin_AM_1001M = time.time() - start_AM_1001M

                print("EL vector solucion es:")
                print(v_sol_AM_1001M)
                print("TIEMPO ", start_fin_AM_1001M)

                vector_AM_1001M = (desv_AM_1001M, valor_inf_AM_1001M, valor_fobjetivo_AM_1001M, start_fin_AM_1001M)
            

                matrix_AGG_UN.append(vector_AGG_UN)
                matrix_AGG_SF.append(vector_AGG_SF)
                #matrix_AGE_UN.append(vector_AGE_UN)
                #matrix_AGE_SF.append(vector_AGE_SF)
                matrix_AM_101.append(vector_AM_101)
                matrix_AM_1001.append(vector_AM_1001)
                matrix_AM_1001M.append(vector_AM_1001M)


            
            nombre_AGG_UN = "datos_AGG_UN_" + f + "_" + f2 + ".xls"
            nombre_AGG_SF = "datos_AGG_SF_" + f + "_" + f2 + ".xls"
            #nombre_AGE_UN = "datos_AGE_UN_" + f + "_" + f2 + ".xls"
            #nombre_AGE_SF = "datos_AGE_SF_" + f + "_" + f2 + ".xls"
            nombre_AM_101 = "datos_AM_101_" + f + "_" + f2 + ".xls"
            nombre_AM_1001 = "datos_AM_1001_" + f + "_" + f2 + ".xls"
            nombre_AM_1001M = "datos_AM_1001MEJOR_" + f + "_" + f2 + ".xls" 
            
            dataframe = pd.DataFrame(matrix_AGG_UN)
            dataframe.to_excel(nombre_AGG_UN)

            dataframe = pd.DataFrame(matrix_AGG_SF)
            dataframe.to_excel(nombre_AGG_SF)

            #dataframe = pd.DataFrame(matrix_AGE_UN)
            #dataframe.to_excel(nombre_AGE_UN)

            #dataframe = pd.DataFrame(matrix_AGE_SF)
            #dataframe.to_excel(nombre_AGE_SF)

            dataframe = pd.DataFrame(matrix_AM_101)
            dataframe.to_excel(nombre_AM_101)

            dataframe = pd.DataFrame(matrix_AM_1001)
            dataframe.to_excel(nombre_AM_1001)

            dataframe = pd.DataFrame(matrix_AM_1001M)
            dataframe.to_excel(nombre_AM_1001M)





"""
f = "iris_set.dat"
f2 = "iris_set_const_10.const"
k = 3
datos, restricciones, n, d = cargar_datos(f, f2)

print("ALGORITMO GENETICO")

long_poblacion = 50
elitista = True
coste_uniforme = True
prob_cruce = 0.7
prob_muta = 0.01
v_sol, desv, valor_inf, valor_fobjetivo = AG(datos, restricciones, n,d,long_poblacion,prob_cruce,prob_muta)

print("EL vector solucion es: ")
print(v_sol)
print("Valor deviacion ", desv)
print("Valor infeasibility ", valor_inf)
print("Valor funcion objetivo ", valor_fobjetivo)

"""
print("ALGORITMO MEMETICO")
long_poblacion = 10
generaciones = 10
tasa = 0.1
tipo_mej = True
prob_cruce = 0.7
prob_muta = 0.01

elitista = True
coste_uniforme = False

v_sol, desv, valor_inf, valor_fobjetivo = AM(generaciones, tasa, tipo_mej, datos, restricciones,n,d,long_poblacion,prob_cruce,prob_muta)

print("EL vector solucion es: ")
print(v_sol)
print("Valor deviacion ", desv)
print("Valor infeasibility ", valor_inf)
print("Valor funcion objetivo ", valor_fobjetivo)
"""
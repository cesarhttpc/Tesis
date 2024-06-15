import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, norm
import time
import os
from pytwalk import BUQ


class VecinosCercanos:

    def __init__(self,puntos):
        # Construir el árbol cKDTree con los puntos
        self.arbol = cKDTree(puntos)
        self.solutions = {}  # Dictionary to store precomputed solutions

    def compute_solutions(self, t, puntos_malla):
        for i, punto in enumerate(puntos_malla):
            solution = odeint(dinamica, y0, t, args=(punto[0], punto[1]))
            self.solutions[i] = solution[:, 0]

    def encontrar_vecinos_cercanos(self, punto, numero_de_vecinos=1):
        # Buscar los vecinos más cercanos del punto dado
        distancias, indices = self.arbol.query(punto, k=numero_de_vecinos)

        # Devolver las distancias y los índices de los vecinos más cercanos
        return distancias, indices
    
def SIR(y,t,beta,gamma):
    I, S, R= y
    dSdt = -beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    return [dIdt, dSdt, dRdt]

dinamica = SIR

y0 = [50, 500, 5]


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


class VecinosCercanos:

    def __init__(self,puntos):
        # Construir el árbol cKDTree con los puntos
        self.arbol = cKDTree(puntos)
        # self.vecinos = num_vecinos

    def encontrar_vecinos_cercanos(self, punto, numero_de_vecinos = 1):
        # Buscar los vecinos más cercanos del punto dado
        distancias, indices = self.arbol.query(punto, k=numero_de_vecinos)

        # Devolver las distancias y los índices de los vecinos más cercanos
        return distancias, indices
    

# Cantidad de vecinos
num_vecinos = 5

# Definir el dominio de los puntos
g_dom = np.linspace(0, 6, num=13)
b_dom = np.linspace(0, 5, num=11)

# Crear la malla utilizando meshgrid
g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)

# Obtener los puntos de la malla y combinarlos en una matriz
puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

# Crear una instancia de la clase VecinosCercanos
buscador_de_vecinos = VecinosCercanos(puntos_malla)

# Punto arbitrario para encontrar vecinos cercanos
punto_arbitrario = np.array([2, 2]) #np.array([2.05, 2.88])  

# Encuentra los vecinos más cercanos al punto arbitrario
distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto_arbitrario, numero_de_vecinos=num_vecinos)

# Imprime los resultados
print("Distancias:", distancias)
print("Índices de vecinos más cercanos:", indices)

# Gráfica puntos en malla
plt.title('Enmallado')
plt.xlabel('g')
plt.ylabel('b')
plt.scatter(g_mesh,b_mesh)
plt.scatter(punto_arbitrario[0],punto_arbitrario[1])

for k in range(num_vecinos):

    print(puntos_malla[indices[k]])
    plt.scatter(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1], color = 'red')
plt.show()







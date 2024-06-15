# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, uniform



class VecinosCercanos:

    def __init__(self,puntos):
        
        # Construir el árbol cKDTree con los puntos
        self.arbol = cKDTree(puntos)

    def compute_solutions(self, t, puntos_malla):
        for i, punto in enumerate(puntos_malla):
            solution = odeint(dinamica, y0, t, args=(punto[0], punto[1]))
            self.solutions[i] = solution[:, 0]

    def encontrar_vecinos_cercanos(self, punto, numero_de_vecinos=1):
        # Buscar los vecinos más cercanos del punto dado
        distancias, indices = self.arbol.query(punto, k=numero_de_vecinos)

        # Devolver las distancias y los índices de los vecinos más cercanos
        return distancias, indices
    

def dinamica(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]


def interpolador(punto, puntos_malla ,t, num_vecinos = 5):

    # Crear una instancia de la clase VecinosCercanos
    buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # Encuentra los vecinos más cercanos al punto arbitrario
    distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto, numero_de_vecinos=5)

    # Pesos
    epsilon = 10**(-6)
    pesos = np.zeros(num_vecinos)
    for i in range(num_vecinos):
         pesos[i] = 1/(distancias[i] + epsilon) 
    norma = sum(pesos)
    pesos = pesos/norma

    n = len(t)
    interpolacion = np.zeros(len(t))
    for k in range(5):
         
        solution = odeint(dinamica, y0, t, args=(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1] ))
        x = solution[:,0]

        interpolacion = interpolacion + x*pesos[k]

    return interpolacion


def logposterior(g, b, t, x, sigma = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
    
        if g>0 and b>0:
            # solution = odeint(dinamica, y0, t, args=(g,b))
            # x_theta = solution[:,0]
            punto = np.array([g,b])
            x_theta = interpolador(punto,puntos_malla, t)
            Logf_post = -n*np.log(sigma) - np.sum((x-x_theta)**2) /(2*sigma**2) + (alpha -1)*np.log(g) - alpha*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

            return Logf_post
        else:
            Logf_post = -10**100
            return Logf_post 
        

def MetropolisHastingsRW(t_datos,x_datos,inicio,size= 100000,alpha= 100, beta= 10, g_0= 10, b_0= 1 ):

        # Punto inicial (parametros)
        x = inicio

        sigma1, sigma2 = 0.3, 0.1

        # 
        sample = np.zeros([size,3])
        sample[0,0] = x[0]  
        sample[0,1] = x[1]
        sample[0,2] = logposterior(x[0], x[1], t_datos, x_datos, alpha=alpha, beta = beta, g_0 = g_0, b_0 = b_0)

        for k in range(size-1):

            # Simulacion de propuesta
            e1 = norm.rvs(0,sigma1)
            e2 = norm.rvs(0,sigma2)
            e = np.array([e1,e2])
            y = x + e   

            # Cadena de Markov
            log_y = logposterior(y[0], y[1], t_datos, x_datos, alpha = alpha, beta = beta, g_0 = g_0, b_0 = b_0)
            log_x = sample[k,2] # Recicla logverosimilitud
            cociente = np.exp( log_y - log_x )

            # Transicion de la cadena
            if uniform.rvs(0,1) <= cociente:

                sample[k+1,0] = y[0]
                sample[k+1,1] = y[1] 
                sample[k+1,2] = log_y

                x = y
            else:
                
                sample[k+1,0] = x[0]
                sample[k+1,1] = x[1] 
                sample[k+1,2] = log_x

        return sample





# if __name__ == "__main__":
   

# Cantidad de vecinos
num_vecinos = 5

# Definir el dominio de los puntos
g_dom = np.linspace(0.01, 0.1, num=15)
b_dom = np.linspace(0.3, 0.6, num=15)

# Crear la malla utilizando meshgrid
g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)

# Obtener los puntos de la malla y combinarlos en una matriz
puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

# Crear una instancia de la clase VecinosCercanos
buscador_de_vecinos = VecinosCercanos(puntos_malla)

# Punto arbitrario para encontrar vecinos cercanos
punto_arbitrario = np.array([0.037, 0.43])  

# Encuentra los vecinos más cercanos al punto arbitrario
distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto_arbitrario, numero_de_vecinos=num_vecinos)

# Imprime los resultados
print("Distancias:", distancias)
print("Índices de vecinos más cercanos:", indices)

# Gráfica puntos en malla
# plt.title('Enmallado')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\gamma$')
plt.scatter(g_mesh,b_mesh)
plt.scatter(punto_arbitrario[0],punto_arbitrario[1], color ='black')

for k in range(num_vecinos):

    print(puntos_malla[indices[k]])
    plt.scatter(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1], color = 'red')
    
plt.savefig('Exp_Central_SIR_sigma/Figuras/Generales/Vecinos_.png', dpi=600, bbox_inches='tight')
plt.show()

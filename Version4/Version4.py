# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma


class VecinosCercanos:

    def __init__(self,puntos):
        # Construir el árbol cKDTree con los puntos
        self.arbol = cKDTree(puntos)

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


def interpolador(punto, puntos_malla ,t):

    # Crear una instancia de la clase VecinosCercanos
    buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # Encuentra los vecinos más cercanos al punto arbitrario
    distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto, numero_de_vecinos=5)

    pesos = np.array([0.6, 0.1, 0.1, 0.1, 0.1]) ##Falta implementar

    n = len(t)
    forwards = np.zeros((n,5))
    interpolacion = np.zeros(len(t))
    for k in range(5):
         
        solution = odeint(dinamica, y0, t, args=(puntos_malla[indices[k][0]],puntos_malla[indices[k][1]] ))

        forwards[:][k] = solution[:,0]

    for i in range(n):

        interpolacion[i] = np.mean(forwards[i,:])
    
    return interpolacion




def logposterior(g, b, t, x, sigma = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
    
        if g>0 and b>0:
            solution = odeint(dinamica, y0, t, args=(g,b))
            x_theta = solution[:,0]
            # punto = (g,b)
            # x_theta = interpolador(punto,puntos_malla, t)
            Logf_post = -n*np.log(sigma) - np.sum((x-x_theta)**2) /(2*sigma**2) + (alpha -1)*np.log(g) - alpha*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

            return Logf_post
        else:
            Logf_post = -10**100
            return Logf_post 
        
        




def MetropolisHastingsRW(t_datos,x_datos,inicio, size = 100000 ,alpha =100, beta = 10, g_0 = 10, b_0 = 1 ):

        # Punto inicial (parametros)
        x = inicio

        sigma1, sigma2 = 0.3, 0.05

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
            # ////////////////////////
            # //////////////////////// se reescribe x ????????
            # ////////////////////////

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


def forwardApprox(punto,puntos_malla):

    # # Crear una instancia de la clase VecinosCercanos
    # buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # # Encuentra los vecinos más cercanos al punto arbitrario
    # distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto, numero_de_vecinos=5)
    
    # pesos = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    # for k in range(5):
         
    #      aux = 0
    #      solution = odeint(dinamica, y0, t, args=(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1] ))

    pass



if __name__ == "__main__":
   

    # Definir el dominio de los puntos
    g_dom = np.linspace(0, 5, num=6)
    b_dom = np.linspace(0, 5, num=6)

    # Crear la malla utilizando meshgrid
    g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)

    # Obtener los puntos de la malla y combinarlos en una matriz
    puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

    # Crear una instancia de la clase VecinosCercanos
    buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # Punto arbitrario para encontrar vecinos cercanos
    punto_arbitrario = np.array([2.05, 2.88])  

    # Encuentra los vecinos más cercanos al punto arbitrario
    distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto_arbitrario, numero_de_vecinos=5)

    # Imprime los resultados
    print("Distancias:", distancias)
    print("Índices de vecinos más cercanos:", indices)

    # Gráfica puntos en malla
    plt.title('Enmallado')
    plt.xlabel('g')
    plt.ylabel('b')
    plt.scatter(g_mesh,b_mesh)
    plt.scatter(punto_arbitrario[0],punto_arbitrario[1])

    for k in range(5):

        print(puntos_malla[indices[k]])
        plt.scatter(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1], color = 'red')
    plt.show()

    #######################################
    ####### Inferencia ####################


    # Parametros principales (verdaderos)
    g = 5.34
    b = 1.15

    # Simular los tiempos de observación
    from scipy.stats import uniform
    n = 31      # Tamaño de muestra (n-1)
    cota = 8
    t = np.linspace(0,cota,num = n)

    # ECUACIÓN DIFERENCIAL:
    # Condiciones iniciales (posición, velocidad)
    y0 = [0.0, 0.0]  

    # Soluciones de la ecuación dínamica
    solutions = odeint(dinamica, y0 ,t, args=(g,b))

    # Coordenadas de caída amortiguada
    x = solutions[:,0]
    v = solutions[:,1]

    # Añadir ruido a los datos
    from scipy.stats import norm
    error = norm.rvs(0,0.01,n)
    error[0] = 0
    x = x + error

    # Grafica
    plt.title('Datos simulados con k = %2.2f , b = %2.2f ' % (g, b))
    plt.xlabel('Tiempo')
    plt.ylabel('Posición')
    plt.scatter(t,x, color= 'orange')
    plt.show()

    
    
    #### MCMC propio ########

    inicio = np.array([8,3])

    # Parametros de distribucion a prioi
    g_0 = 5
    alpha = 1
    b_0 = 2
    beta = 1.5

    sample = MetropolisHastingsRW(t, x, inicio, size = 60000, g_0 = g_0, b_0 = b_0, beta = beta, alpha= alpha)

    #Visualización
    g_sample = sample[:,0]
    b_sample = sample[:,1]
    log_post = sample[:,2]

    burn_in = 5000

    plt.title('Cadena')
    plt.plot(g_sample[:10000],label = 'g')
    plt.plot(b_sample[:10000],label = 'b')
    plt.legend()
    plt.show()

    plt.title('Trayectoria de MCMC')
    plt.plot(g_sample,b_sample,linewidth = .5, color = 'gray')
    plt.xlabel('g')
    plt.ylabel('b')
    plt.show() 

    plt.title('LogPosterior de la cadena')
    plt.plot(log_post, color = 'red')
    plt.show()

    plt.title('Distribuciones a priori y posterior para g')
    plt.hist(g_sample[burn_in:], density= True, bins = 40)
    dom_g = np.linspace(0,15,500)
    plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = g_0/alpha),color = 'green')
    plt.ylabel(r'$f(g)$')
    plt.xlabel(r'$g$')
    linea = np.linspace(0,0.5, 100)
    linea_x = np.ones(100)
    plt.plot(linea_x*g, linea, color = 'black', linewidth = 0.5)
    plt.show()

    plt.title('Distribuciones a priori y posterior para b')
    plt.hist(b_sample[burn_in:], density= True, bins = 40)
    dom_b = np.linspace(0,8,500)
    plt.plot(dom_b, gamma.pdf(dom_b, a = beta, scale = b_0/beta),color = 'green')
    plt.ylabel(r'$f(b)$')
    plt.xlabel(r'$b$')
    linea = np.linspace(0,1, 100)
    linea_x = np.ones(100)
    plt.plot(linea_x*b, linea, color = 'black', linewidth = 0.5)
    plt.show()











# %%
    
solution = odeint(dinamica, y0, t, args=(8,1))

x_theta = solution[:,0]

print(x_theta)
print(t)



print(len(t))

# %%
n = 5
z = np.zeros((n,4))
for j in range(5):
    for k in range(4):
        z[j,k] = 3*k-2 + j*(1-j)
print(z)

for i in range(n):
    print(np.mean(z[i,:]))
    
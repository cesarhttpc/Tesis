# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def libre(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g
    return [dxdt, dvdt]

def damping(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def resorte(y,t,k,b):
    x, v = y
    dxdt = v
    dvdt = -k*x - b*v
    return [dxdt, dvdt]

def forward(g,b):
    
    solutions = odeint(damping, y0, t, args=(g,b))
    return solutions[:,0]


y0 = [0, 0.0]  
g = 10
b = .7
t = np.linspace(0, .5, 1000)  

# Soluciones de la ecuación dínamica
solutions = odeint(damping, y0, t, args=(g,b))

solutions0 = odeint(libre, y0, t, args=(g,b))
k = 1
solutions1 = odeint(resorte,y0, t, args=(k,b))

x = solutions[:, 0]
v = solutions[:, 1]


# Graficas
plt.plot(t, x, label='Caida amortiguada')
# plt.plot(t, v, label='Velocidad v')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Mecánica Clasica')
plt.plot(t, solutions0[:,0], label = 'Caída libre')
plt.legend()
# plt.show()

# z = lambda x: 9.7*x**2/2
# plt.plot(t,z(t))
 


y_o = np.array([0, 0.06 ,0.16, 0.26, 0.36, 0.46, 0.56])
t_o = np.array([0, 0.071, 0.158, 0.220, 0.267, 0.306, 0.340])

plt.plot( t_o, y_o, 'o')
plt.show()


plt.plot(x,v)
plt.title('Diagrama de fase')
plt.xlabel('Pocisión')
plt.ylabel('Velocidad')
plt.legend()
plt.show()



# %%
# def posterior0(g,b,sigma,y,t):

#     n = len(y)
#     h = 0.5/1000
#     solucion = odeint(damping, y0, t, args=(g,b))
#     x = solucion[:,0]

#     cuadrados = np.zeros(n)
#     for j in range(n):
#         cuadrados[j] = (y[j] - x[int(t[j]/h)])**2

#     f_post = (1/sigma)**(n/2) * np.exp(- sum( cuadrados)/(2*sigma**2))* np.exp(-(g-10)**2/(0.18))* b**9* np.exp(-10*b)

#     return f_post





def posterior(g,b,y,t,sigma = 1):
    
    n = len(y)
    h = 0.5/1000    
    solucion = odeint(damping, y0, t, args=(g,b))
    x = solucion[:,0]

    cuadrados = np.zeros(n)
    for j in range(n):
        print(j)    
        print(t_o[j],'   ',int(t[j]/h))
        cuadrados[j] = (y[j]- x[int(t[j]/h)])**2

        Logf_post = n*np.log(sigma) - sum(cuadrados)/(2*sigma**2) + (alpha -1)*np.log(g) + 9*np.log(b) - alpha*g/10 - 10 * b


alpha = 1000

print(posterior(10,1,y_o,t_o))


# %%





# def posteriorlibre(g,sigma,y):

#     n = len(y) 

#     y
#     f_post = (1/sigma)**(n/2) * np.exp(- )


# %%
    
print(x[int()])




















# %%
from scipy.stats import gamma , norm

# Distribuciones a priori
dom = np.linspace(0,4,500)
dom1 = np.linspace(0,15,500)

plt.plot(dom, gamma.pdf(dom,a = 10, scale = .1))  # 5, 0.1
plt.title('priori para b ')
plt.show()

alpha = 1000
plt.plot(dom1, gamma.pdf(dom1, a = alpha , scale = 9.81/alpha))
plt.show()

plt.plot(dom1, norm.pdf(dom1, 10, 0.3))
plt.title('Priori para g')
plt.show()


# %%















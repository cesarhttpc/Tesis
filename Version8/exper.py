# %%
import numpy as np
from scipy.integrate import odeint  
import matplotlib.pyplot as plt 


def logistico(P,t,K,r):

    dPdt = r*P*(1-P/K)
    return dPdt

#Cond. inicial
y0 = [6.0]

# Parametros
r = 1.21
K = 1325

t = np.linspace(0,10)

P = odeint(logistico,y0,t, args=(K,r))

x = P[:,0]
plt.plot(t,x)

# %%

def gravedad(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def Forward( theta, t):

    g,b = theta

    solutions = odeint(dinamica, y0 ,t, args=(g,b))
    x = solutions[:,0]
    return x

def gravedad_analitica(t,g,b):

    m = 1
    v_Terminal = m*g/b

    x = v_Terminal*(t - m*(1-np.exp(-b*t/m))/b)
    return x



dinamica = gravedad
y0 = [0.0, 0.0]

g = 9.8
b = 5.22
theta = [g,b]

t = np.linspace(0,2,100)
# F = Forward(theta, t)
# plt.plot(t,F)

plt.title('Trayectorias modelo gravitatorio')
g = 9.8
b = 5.22
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 7.8
b = 2.0
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 15
b = 9.3
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 2.8
b = 0.4
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend()



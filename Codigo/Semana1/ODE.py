# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
 
def coupled_differential_equations(y, t, k, b):
    x, v = y
    dxdt = v
    dvdt = -k * x - b * v
    return [dxdt, dvdt]
 
y0 = [1.0, 0.0]  
k = 1.0 
b = 0.01
t = np.linspace(0, 30, 1000)  
 
solutions = odeint(coupled_differential_equations, y0, t, args=(k, b))
 
x_values = solutions[:, 0]
v_values = solutions[:, 1]
 
plt.plot(t, x_values, label='Position x')
plt.plot(t, v_values, label='Velocity v')
plt.xlabel('Time')
plt.ylabel('Position / Velocity')
plt.legend()
plt.title('Coupled Differential Equations: Simple Harmonic Oscillator with Damping')
plt.show()


plt.plot(x_values,v_values)
plt.plot()


# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t):
    k = 0.3
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title('Simple Harmonic Oscillator with Damping')
plt.show()


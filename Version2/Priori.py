# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
dom_g = np.linspace(0,15,500)
dom_b = np.linspace(0,4,500)

alpha = 100
plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = 10/alpha))
plt.title('Distribución a priori de g')
plt.ylabel(r'$f(g)$')
plt.xlabel(r'$g$')
plt.show()

plt.plot(dom_b, gamma.pdf(dom_b, a = 10, scale = .1))  # 5, 0.1
plt.title('Distribución a priori de b ')
plt.ylabel(r'$f(b)$')
plt.xlabel(r'$b$')
plt.show()


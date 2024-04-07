# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)

log = norm.logpdf(x, loc = 0, scale = 3)
plt.plot(x,log)
plt.show()
plt.plot(x,np.exp(log))


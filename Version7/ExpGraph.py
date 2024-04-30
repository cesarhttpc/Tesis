# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

Monte_carlo_todo = {}

for i in range(4):
    Monte_carlo_todo[i] = norm.rvs(3*i,1,25*i+100)



# for k in range(3):
#     plt.hist(Monte_carlo_todo[k], density= True, alpha = 0.5)
fig, axs = plt.subplots(1,3,figsize = (8,5))
fig.suptitle('Vertically stacked subplots')
axs[0].hist(Monte_carlo_todo[0], density = True, histtype = 'step')
axs[1].hist(Monte_carlo_todo[1], density = True, histtype = 'barstacked')
axs[2].hist(Monte_carlo_todo[2], density = True)
# axs[0].label_outer()
axs[1].label_outer()
axs[2].label_outer()
plt.figure(figsize=(5,35))
plt.show()

# %%


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Horizontally stacked subplots')
ax1.hist(Monte_carlo_todo[0], density = True)
ax2.hist(Monte_carlo_todo[1], density = True)
ax3.hist(Monte_carlo_todo[2], density = True)
plt.show()



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.hist(Monte_carlo_todo[0], density = True)
ax2.hist(Monte_carlo_todo[1], density = True)
ax3.hist(Monte_carlo_todo[2], density = True)
ax4.hist(Monte_carlo_todo[3], density = True)

for ax in fig.get_axes():
    ax.label_outer()


# fig, axs = plt.subplots(2,2)
# axs[0].hist(Monte_carlo_todo[0],density = True)
# axs[1].hist(Monte_carlo_todo[1],density = True)
# axs[2].hist(Monte_carlo_todo[2],density = True)
# axs[3].hist(Monte_carlo_todo[3],density = True)

# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

m = 3000
estimador = np.zeros(m) 
estimador_sesgo = np.zeros(m)

# plt.figure(figsize=(5,10))
for k in range(m):

    z = norm.rvs(0,scale = 7,size= 1000)
    mean = np.mean(z)
    n = len(z)

    suma = np.sum((z - mean)**2)/(n-1)
    suma_sesgo = suma * (n-1) / n
    estimador[k] = suma
    estimador_sesgo[k] = suma_sesgo
plt.hist(estimador, alpha = 0.8, histtype='step')
plt.show()
# plt.hist(estimador_sesgo, alpha = 0.5)

# print(suma)
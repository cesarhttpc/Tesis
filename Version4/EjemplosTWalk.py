# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, gamma




from pytwalk import pytwalk, pyPstwalk, Ind1Dsampl, BUQ

# %%
if __name__ == "__main__":
    ### Example using derived class BUQ
    ### Define the Forward map with signature F( theta, t)
    def F( theta, t):
        """Simple analytic FM, for this example.  Logistic function."""

        L, k, t0 = theta
        return L/(1 + np.exp(-k*(t-t0)))


    t = np.linspace(-2, 7, num=30) #The sample size is 30, a grid of size 30
    sigma = 0.1 #Stardard dev. for data
    ### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
    ### see docstring of BUQ
    logdensity=norm.logpdf
    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[        r"$L$", r"$k$", r"$t_0$"]
    par_prior=[ gamma( 3, scale=1), gamma(1.1, scale=1), norm(loc=0,scale=1)]
    par_supp  = [ lambda L: L>0.0, lambda k: k>0.0, lambda t0: True]
    #data = array([3.80951951, 3.94018984, 3.98167993, 3.93859411, 4.10960395])
    buq = BUQ( q=3, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma,\
                F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    buq.SimData(x = np.array([ 1, 1, 0])) #True parameters alpha=3 lambda=0.1
    ### The two initial values are simulated from the prior
    ### redifine buq.SimInit eg.
    ### buq.SimInit = lambda: array([ 0.001, 1000])+norm.rvs(size=3)*0.001

    buq.RunMCMC( T=100_000, burn_in=10000, fnam="buq_output.csv")
    buq.PlotPost(par=0, burn_in=10000)
    buq.PlotPost(par="k", burn_in=10000) #we may acces the parameters by name also
    buq.PlotCorner(burn_in=10000)
    print("The twalk sample is available in buq.Ouput: ", buq.Output.shape)




    ##########################  Forward Map EDO ##############

# %%
    ### Example using derived class BUQ
    ### Define the Forward map with signature F( theta, t)

    def dinamica(y,t,k,b):
        x, v = y
        dxdt = v
        dvdt = -k*x - b*v
        return [dxdt, dvdt]

    def F( theta, t):

        g,b = theta

        y0 = [1.0, 0.0]  
        solutions = odeint(dinamica, y0 ,t, args=(g,b))
        x = solutions[:,0]
        return x


    t = np.linspace(-2, 7, num=30) #The sample size is 30, a grid of size 30
    sigma = 0.1 #Stardard dev. for data
    ### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
    ### see docstring of BUQ
    logdensity=norm.logpdf
    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[  r"$k$", r"$b$" ]   #, r"$t_0$"]
    par_prior=[ gamma( 3, scale=1), gamma(1.1, scale=1)]   # , norm(loc=0,scale=1)]
    par_supp  = [ lambda k: k>0.0, lambda b: b>0.0]   # , lambda t0: True]
    #data = array([3.80951951, 3.94018984, 3.98167993, 3.93859411, 4.10960395])
    buq = BUQ( q=3, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma,\
                F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    buq.SimData(x = np.array([ 4, 2]))#, 0])) #True parameters alpha=3 lambda=0.1
    ### The two initial values are simulated from the prior
    ### redifine buq.SimInit eg.
    ### buq.SimInit = lambda: array([ 0.001, 1000])+norm.rvs(size=3)*0.001

    buq.RunMCMC( T=100_000, burn_in=10000, fnam="buq_output.csv")
    buq.PlotPost(par=0, burn_in=10000)
    buq.PlotPost(par=1, burn_in=10000) #we may acces the parameters by name also
    buq.PlotCorner(burn_in=10000)
    print("The twalk sample is available in buq.Ouput: ", buq.Output.shape)

# %%
    ### Example using derived class BUQ
    ### Define the Forward map with signature F( theta, t)

    def dinamica(y,t,g,b):
        x, v = y
        dxdt = v
        dvdt = g - b*v
        return [dxdt, dvdt]
    


    def F( theta, t):

        g,b = theta

        y0 = [0.0, 0.0]  
        solutions = odeint(dinamica, y0 ,t, args=(g,b))
        x = solutions[:,0]
        return x


    t = np.linspace(-2, 7, num=30) #The sample size is 30, a grid of size 30
    sigma = 0.1 #Stardard dev. for data
    ### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
    ### see docstring of BUQ
    logdensity=norm.logpdf
    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[  r"$g$", r"$b$" ]   #, r"$t_0$"]
    par_prior=[ gamma( 3, scale=1), gamma(1.1, scale=1)]   # , norm(loc=0,scale=1)]
    par_supp  = [ lambda k: k>0.0, lambda b: b>0.0]   # , lambda t0: True]
    #data = array([3.80951951, 3.94018984, 3.98167993, 3.93859411, 4.10960395])
    buq = BUQ( q=2, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma,\
                F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    buq.SimData(x = np.array([ 4, 2]))#, 0])) #True parameters alpha=3 lambda=0.1
    ### The two initial values are simulated from the prior
    ### redifine buq.SimInit eg.
    ### buq.SimInit = lambda: array([ 0.001, 1000])+norm.rvs(size=3)*0.001

    buq.RunMCMC( T=100_000, burn_in=10000, fnam="buq_output.csv")
    buq.PlotPost(par=0, burn_in=10000)
    buq.PlotPost(par="k", burn_in=10000) #we may acces the parameters by name also
    buq.PlotCorner(burn_in=10000)
    print("The twalk sample is available in buq.Ouput: ", buq.Output.shape)










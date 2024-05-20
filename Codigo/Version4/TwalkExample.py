#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#  twalktutorial2.py
#  
#  Examples for the twalk implementation in Python,
#  Created by J Andres Christen, jac at cimat.mx .
#
#  New version, June 2023.  The algorithm is the same, only new working and
#  ploting features have been added.

**************************************************************************************
HIGHLIGHTS OF NEW FEATURES!!!!:
    
0) pytwalk is now installed with pip, as any other Python module.  See the web
page https://www.cimat.mx/~jac/twalk/
    
1) To continue a t-walk run, simply put x0=None, xp0=None in the Run method.
It will use the last x and xp points of the previous run, and continue with
as many iterations as wanted.  These will be added to Output etc. to make it as
one single run.   See "Independent Normals" example below.

2) To stop a run, use ctrl-C in the running terminal.  The current iteration
will be finished and Output etc will have only the number of saved iterations
so far.

3) All plotting methods now use subplots(), and return the Axes being used, for
further customization by the user.  Also, all plotting methods receive an
optional Axes object in the argument ax, to have the plotting method use it to
plot: see "Bernoulli Regression" example.

4) Now use "burn_in" argument, in all methods, instead of "start" (normally the
latter is kept for legacy reasons).
                                                                   
5) For Bayesian inference, the template class pyPstwalk has been added, see the
Berreg example.  There is also the simplified Ind1Dsampl class, for independent
1D sampling, see the "Gamma sampling" example. Also the BUQ class has been
added for Bayesian Uncertainty Quantification (non-linear regression).
See example "BUQ".

6) A run may be saved with SavetwalkOutput and retrieved with LoadtwalkOutput,
see the "Related Bernoulli trials" example.

7) Parameters names can now be used
(e.g par_names = [ r"$\theta_0$", r"$\theta_1$", r"$\theta_2$"]).  For the
pyPstwalk derived classes parameters may also be accessed by name,
e.g. gammasmpl.PlotPost(par="beta", density=True) .
See the "Gamma sampling" example.

8) There are other minor features added, see all the examples below.
                                                                   
BUT!!!  The t-walk algorithm is *the same* as in previous version and
IT IS BACKWARDS COMPATIBLE for the Python 3 version.  All your programs should
run the same.

Have fun!  Andres Christen
22 JUN 2023

****************************************************************************************



"""
# %%

from numpy import ones, zeros, log, array, exp, linspace
from numpy import sum as np_sum
from scipy.stats import uniform, bernoulli, weibull_min, norm, gamma, expon
from matplotlib.pylab import subplots

from pytwalk import pytwalk, pyPstwalk, Ind1Dsampl, BUQ

# %%

class Berreg(pyPstwalk):
    
    """ 'Bernoulli regresion' example dereived class from pyPstwalk
        for Bernoulli regression:
        $$
        X_{i,j} | \theta, a, b  \sim Ber( \Phi_{\theta} ( a + t_j b))
        $$
        where $\Phi$ is a pdf, default Weibull.
        
        t: array of lenght $m$ with times $t_j$
        data: nxm array with data.
    """
    
    def __init__( self, t, data, dist=weibull_min):

        self.t = t
        self.m = t.size
        self.data = data
        self.dist = dist
        self.n = self.data.shape[0]
        
        super().__init__(par_names=[              r"$a$",             r"$b$",        r"$\alpha$",       r"$\beta$"],\
                         par_prior=[norm(loc=0, scale=1), gamma( 3, scale=1), gamma( 3, scale=1), gamma( 3, scale=1)],\
                         par_supp =[      lambda x: True,    lambda x: x>0.0,    lambda x: x>0.0,   lambda x: x>0.0])
            
    def loglikelihood(self, x):
        a, b, al, beta = x
        ll = 0.0
        for i in range(self.n):
            ll += np_sum( bernoulli.logpmf( self.data[i,:],\
                                self.dist.cdf( a+b*self.t, al, scale=beta)))
        return ll


if __name__ == "__main__":
    
    # To run all examples:
    ex_list = ["Independent Normals",
               "Product of Exponentials",
               "Related Benoulli trials",
               "Gamma sampling",
               "BUQ",
               "Bernoulli Regresion"]
    # ex_list = ["BUQ"] #"Independent Normals","Product of Exponentials", "Gamma sampling", "BUQ", "Bernoulli Regresion"]

    if "Independent Normals" in ex_list:
        """
        ########################################################
        ### This sets a twalk for n independent normals ########
        ### Three basic ingredients are needed:
        ### 1) define -log of the objective (posterior)
        ### 2) define the support function, return True if parameters are in the support
        ### 3) Define *two* initial points for the t-walk.
        ### Each parameter needs to be different in the initial points.
        ### Commonly, we include a function to simulae two random points, within the support is used.
        
        ### Two initial points remotely within the same 'galaxy'
        ### as the actual 'effective' support
        ### (where nearly all of the probability is) of the objective, are needed.
        """
        
        def NormU(x):
            """Defines the 'Energy', -log of the objective, besides an arbitrary constant."""
            return sum(0.5*x**2)
        
        def NormSupp(x):
            return True
        
        nor = pytwalk( n=10, U=NormU, Supp=NormSupp)
        
        ### This runs the twalk, with initial points 10*ones(10) and zeros(10)
        ### with T=100000 iterations
        print("\n\nRunning the dimension 10 Gaussian example:""")
        nor.Run( T=100000, x0=10*ones(10), xp0=zeros(10))
        
        ### This does a basic output analysis, with burnin=1000, in particular the IAT:
        iat = nor.Ana(burn_in=1000)
        nor.PlotMarg( par=0, burn_in=1000)
        nor.PlotCorner( pars=[0,1,2], burn_in=1000)
        ### A new run may be done ... here we save only every thin itereations
        ### ie, online thining, this is done normally for really big samples
        ### in big dimensions to save memory space, only.
        nor.Run( T=100000, x0=10*ones(10), xp0=zeros(10), thin=iat)
        ### To continue the same run, adding more iterations do
        nor.Run( T=100000, x0=None, xp0=None, thin=iat)
        print("Sample size= %d" % (nor.T))

    if "Product of Exponentials" in ex_list:
        """
        ########################################################
        ######### Example of a product of exponentilas #########
        ### Objective function:
        ### $f(x_1,x_2,x_3,x_4,x_5)=\prod_{i=1}^5 \lambda exp(-x_i \lambda)$
        """
        
        lambdas = [ 1., 2., 3., 4., 5.]
        
        def ExpU(x):
            # """-log of a product of exponentials"""
            return sum(x * lambdas)
    
        def ExpSupp(x):
        	return all(0 < x)
        
        Exp = pytwalk( n=5, U=ExpU, Supp=ExpSupp)
        
        ### This runs the twalk, with initial points ones(40) and 40*ones(40)
        ### with T=50000 iterations
        print("\n\nRunning the product of exponentials example:""")
        Exp.Run( T=50000, x0=30*ones(5), xp0=40*ones(5))
        Exp.Ana(burn_in=2000)
        Exp.PlotMarg( par=2, burn_in=2000) # Plot the third parameter


    if "Related Benoulli trials" in ex_list:
        """
        #########################################################################
        ##### A more complex example:
        ##### Related Bernoulli trials #####
        ##### Suppose x_{i,j} ~ Be( theta_j ), i=0,1,2,...,n_j-1, ind. j=0,1,2
        ##### But it is known that 0 <  theta_0 < theta_2 < theta_3 < 1 
        """
        
        theta = array([ 0.4, 0.5, 0.7 ])  ### True thetas
        n = array([ 20, 15, 40]) ### sample sizes
        #### Simulated synthetic data: 
        r = zeros(3)
        for j in range(3):
            ### This su¡imulates each Bernoulli: uniform.rvs(size=n[j]) < theta[j]
            ### but we only need the sum of 1's (sum of True's)
        	#r[j] = sum(uniform.rvs(size=n[j]) < theta[j])
            r = array([ 9,  9, 28])
        
        ### Defines the support.  This is basically the prior, uniform in this support
        def ReBeSupp(theta):
        	rt = True
        	rt &= (0 < theta[0])
        	rt &= (theta[0] < theta[1])
        	rt &= (theta[1] < theta[2])
        	rt &= (theta[2] < 1)
        	return rt
        
        #### It is a good idea to have a function that produces random initial points,
        #### indeed always withinh the support
        #### In this case we simulate from something similar to the prior
        def ReBeInit():
        	theta = zeros(3)
        	theta[0] = uniform.rvs()
        	theta[1] = uniform.rvs( loc=theta[0], scale=1-theta[0])
        	theta[2] = uniform.rvs( loc=theta[1], scale=1-theta[1])
        	return theta
        
        ####### The function U (Energy): -log of the posterior
        def ReBeU(theta):
        	return -1*sum(r * log(theta) + (n-r)*log(1.0-theta))
        
        ###### Define the twalk instance
        ReBe = pytwalk( n=3, U=ReBeU, Supp=ReBeSupp)
        ReBe.par_names = [ r"$\theta_0$", r"$\theta_1$", r"$\theta_2$"]        
        #### This runs the twalk
        print("\n\nRunning the related Bernoullis example:""")
        ReBe.Run( T=100000, x0=ReBeInit(), xp0=ReBeInit())
        
        ### First plot a trace of the log-post, to identify the burn-in
        ReBe.PlotTs() 
        
        ### This will do a basic output analysis, with burnin=5000
        ReBe.Ana(burn_in=5000)
        #ReBe.PlotTs(par=0, burn_in=5000) # Trace plot of par 0
        
        ### And some histograms
        ReBe.PlotMarg( par=0, burn_in=5000 )
        ReBe.PlotMarg( par=1, burn_in=5000 )
        ReBe.PlotMarg( par=2, burn_in=5000 )	
        ReBe.PlotCorner(burn_in=5000)
        
        ### Then save it to a text file, with each column for each paramter
        ### plus the U's in the last column, that is T+1 rows and n+1 colums.
        ### This may in turn be loaded by other programs
        ### for more sophisticated output analysis (eg. BOA).
        ReBe.SavetwalkOutput("RelatedBer.csv", burn_in=5000)
        
        ### You may access the (T+1) X (n+1) output matrix directly with
        #ReBe.Output
        
        ### To retirve a run use
        #ReBe.LoadtwalkOutput("RelatedBer.csv")
        
        ###### All methods should have help lines: ReBe.Hist?

        
    
    if "Gamma sampling" in ex_list:
        ### Example of using Ind1Dsampl
        ### Example with Gamma data with both shape and *rate* parameters unknown
        q = 2 # Number of unkown parameters, ie dimension of the posterior
        ###                       x[0],     x[1]
        par_names = [       r"$\alpha$", r"$\beta$"] # shape and rate
        ### The piros are assumed independent and exponential
        ### I use 'frozen' distributions from scipy.stats,
        ### although only the methods logpdf and rvs are required
        par_prior = [    expon(scale=20), expon(scale=20)]
        ### The support (both positive) is defined like this:
        par_supp  = [ lambda al: al>0.0, lambda beta: beta>0.0]
        ### This is log of the density, needs to be vectorized, and a function (optional) to simulate data
        logdensity = lambda data, x: gamma.logpdf( data, a=x[0], scale=1/x[1])
        simdata    = lambda n, x: gamma.rvs(a=x[0], scale=1/x[1], size=n)
        gammasmpl = Ind1Dsampl( q=q, data=None, logdensity=logdensity,\
                par_names=par_names, par_prior=par_prior, par_supp=par_supp, simdata=simdata)
        ### Simulate data using par_true = array([ 3, 10]), alpha=3, beta=10
        #gammasmpl.par_true = array([ 3, 10])
        gammasmpl.SimData( n=30, x=array([ 3, 10])) # Simulate data
        gammasmpl.RunMCMC( T=150000, burn_in=5000) #fnam=None, output not saved to file
        ### Rretive the ploting Axes:
        ax_al = gammasmpl.PlotPost(par=0, density=True)
        ### Costumize the plot:
        ax_al.set_ylabel("density")
        ### Access the parameter by name:
        ax_beta = gammasmpl.PlotPost(par="beta", density=True)
        gammasmpl.PlotCorner()
        print("Gamma sampling, sample size %d, true parameters:"\
              % (gammasmpl.smpl_size), gammasmpl.par_true )
    
    if "BUQ" in ex_list:
        ### Example using derived class BUQ
        ### Define the Forward map with signature F( theta, t)
        def F( theta, t):
            """Simple analytic FM, for this example.  Logistic function."""
            L, k, t0 = theta
            return L/(1 + exp(-k*(t-t0)))
        t = linspace(-2, 7, num=30) #The sample size is 30, a grid of size 30
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
        buq.SimData(x=array([ 1, 1, 0])) #True parameters alpha=3 lambda=0.1
        ### The two initial values are simulated from the prior
        ### redifine buq.SimInit eg.
        ### buq.SimInit = lambda: array([ 0.001, 1000])+norm.rvs(size=3)*0.001

        buq.RunMCMC( T=100_000, burn_in=10000, fnam="buq_output.csv")
        buq.PlotPost(par=0, burn_in=10000)
        buq.PlotPost(par="k", burn_in=10000) #we may acces the parameters by name also
        buq.PlotCorner(burn_in=10000)
        print("The twalk sample is available in buq.Ouput: ", buq.Output.shape)
    
    if "Bernoulli Regresion" in ex_list:
        ### Bernoulli regression example using Berreg derived class
        ### See __doc__ of Berreg
        t = array([0, 10, 20 , 30, 40])
        m = t.size
        ### Values to simulate data:
        n = 30
        data = zeros((n,m))
        a=0.5
        b=0.01
        al = 2
        beta = 1
        for i in range(n):
            data[i,:] = bernoulli.rvs( weibull_min.cdf( a+b*t, al, scale=beta))
        print(data)
    
        br = Berreg( t, data)
        br.par_true = array([ a, b, al, beta])
        br.RunMCMC(T=100000, burn_in=0)
        ### Plot all posterior marginals in a panel.
        fig, ax = subplots(nrows=2,ncols=2)
        ax = ax.flatten()
        for i in range(4):
            br.PlotPost( par=i, ax=ax[i], burn_in=1000)
        br.PlotCorner(burn_in=1000)
        """
        ##### Tostadas data:
        data_tostadas = loadtxt("tostadas.csv", skiprows=1)
        t_tostadas = array([0, 10, 20 , 30, 40])
        tostadas = berreg( t, data, defualt_burn_in=1000)
        """

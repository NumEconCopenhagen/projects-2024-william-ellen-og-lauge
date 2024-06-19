

# a. imports
from types import SimpleNamespace
import numpy as np
#import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})
 
# b. imported classes
class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1-par.w1A
        par.w2B = 1-par.w2A

    def utility_A(self,x1A,x2A):
        par = self.par
        utilA = x1A**par.alpha*x2A**(1-par.alpha)
        return utilA
    
    def find_market_clearing_price(self):
        def objective(p1):
            eps1, eps2 = self.check_market_clearing(p1)
            return eps1 + eps2

        p1_initial = 1
        result = fsolve(objective, p1_initial)
        return result[0]
        

    def utility_B(self,x1B,x2B):
        par = self.par
        utilB = x1B**par.beta*x2B**(1-par.beta)
        return utilB

    def demand_A(self,p1):
        par = self.par
        I = p1*par.w1A + par.w2A  # Income of A
        x1A = par.alpha * I / p1  # Demand for good 1
        x2A = (1-par.alpha) * I  # Demand for good 2
        return x1A, x2A

    def demand_B(self,p1):
        par = self.par
        I = p1*par.w1B + par.w2B  # Income of B
        x1B = par.beta * I / p1  # Demand for good 1
        x2B = (1-par.beta) * I  # Demand for good 2
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        ## Market clearing. If the markets clear, the excess demand is zero.
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def optimize_allocation(self):
        par = self.par

        # Compute utility at endowment for B
        util_endowment_B = self.utility_B(par.w1B, par.w2B)
        util_endowment_A = self.utility_A(par.w1A, par.w2A)

        # Define the feasible set C
        N = 75
        feasible_set_C = [(x1A, x2A) for x1A in np.arange(0, N+1, 1) / N
                          for x2A in np.arange(0, N+1, 1) / N
                          if self.utility_B(1 - x1A, 1 - x2A) >= util_endowment_B]

        # Initialize maximum utility and corresponding allocation for A
        max_util_A = util_endowment_A
        optimal_allocation_A = (par.w1A, par.w2A)

        # Iterate over all allocations in the choice set C
        for x1A, x2A in feasible_set_C:
            util_A = self.utility_A(x1A, x2A)
            # Check if this allocation yields a higher utility than the current maximum for A
            if util_A > max_util_A:
                max_util_A = util_A
                optimal_allocation_A = (x1A, x2A)

        return optimal_allocation_A, max_util_A
    

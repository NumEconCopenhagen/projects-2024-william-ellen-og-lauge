

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
    

########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########

def results(economy):
    # Utility at endowment
    endowment_utility_A = economy.utility_A(economy.par.w1A, economy.par.w2A)
    endowment_utility_B = economy.utility_B(economy.par.w1B, economy.par.w2B)
    # Endowments
    endow_A = (economy.par.w1A, economy.par.w2A)
    endow_B = (economy.par.w1B, economy.par.w2B)
    # Optimal allocation
    optimal_allocation_A, max_util_A = economy.optimize_allocation()
    # We assume the final allocation for B is whatever is left after A's allocation
    final_allocation_B = (1-optimal_allocation_A[0], 1-optimal_allocation_A[1])
    # Utility for B at the final allocation
    final_utility_B = economy.utility_B(final_allocation_B[0], final_allocation_B[1])
    
    # Print results
    print(f"Utility for A at A's endowment: {endowment_utility_A}")
    print(f"Utility for B at B's endowment: {endowment_utility_B}")
    print(f"Endowment for A: {endow_A}")
    print(f"Endowment for B: {endow_B}")
    print(f"Optimal allocation for A: {optimal_allocation_A}")
    print(f"Max utility for A at optimal allocation: {max_util_A}")
    print(f"Final allocation for B: {final_allocation_B}")
    print(f"Utility for B at final allocation: {final_utility_B}")
    return optimal_allocation_A, max_util_A

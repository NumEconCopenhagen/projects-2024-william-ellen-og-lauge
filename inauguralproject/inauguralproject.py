

# a. imports
from types import SimpleNamespace
import numpy as np
#import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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

# Testing...
# results = ExchangeEconomyClass().utility_A(1,1)
# print(results)
    
# create an instance of the class
economy = ExchangeEconomyClass()
par = economy.par

# Compute initial utilities
initial_utility_A = economy.utility_A(par.w1A, par.w2A)
initial_utility_B = economy.utility_B(par.w1B, par.w2B)

initial_utility_A, initial_utility_B

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have defined 'economy' and 'initial_utility_A', 'initial_utility_B' somewhere before

# Parameters
N = 75
x1A_values = np.linspace(0, 1, N+1)
x2A_values = np.linspace(0, 1, N+1)

# Create arrays to store Pareto efficient allocations
pareto_efficient_allocations = []

# Iterate over all possible combinations of x1A and x2A
for x1A in x1A_values:
    for x2A in x2A_values:
        # Ensure x1B and x2B are non-negative
        x1B = 1 - x1A
        x2B = 1 - x2A
        if x1B >= 0 and x2B >= 0:
            # Compute utilities for the current allocation
            utility_A = economy.utility_A(x1A, x2A)
            utility_B = economy.utility_B(x1B, x2B)

            # Check if both utilities are at least as high as the initial endowment utilities
            if utility_A >= initial_utility_A and utility_B >= initial_utility_B:
                pareto_efficient_allocations.append((x1A, x2A))

# Convert the list of allocations to a NumPy array for plotting
pareto_efficient_allocations = np.array(pareto_efficient_allocations)

# Plot the Edgeworth box with the Pareto efficient allocations
fig = plt.figure(frameon=False, figsize=(8, 8), dpi=100)
ax_A = fig.add_subplot(1, 1, 1)

ax_A.set_xlabel("$x_1^A$")
ax_A.set_ylabel("$x_2^A$")

# Plot Pareto efficient allocations
ax_A.plot(pareto_efficient_allocations[:, 0], pareto_efficient_allocations[:, 1], 'o', markersize=2, label='Pareto Efficient Allocations')

# Plot initial endowments
ax_A.plot(par.w1A, par.w2A, 'r*', markersize=10, label='Initial Endowment A')
ax_A.plot(par.w1B, par.w2B, 'b*', markersize=10, label='Initial Endowment B')

# A
ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')

# limits
ax_A.plot([0, 1], [0, 0], lw=2, color='black')
ax_A.plot([0, 1], [1, 1], lw=2, color='black')
ax_A.plot([0, 0], [0, 1], lw=2, color='black')
ax_A.plot([1, 1], [0, 1], lw=2, color='black')

ax_A.set_xlim([0, 1])
ax_A.set_ylim([0, 1])
  

# Design for B
temp = ax_A.twinx()
temp.set_ylabel("$x_2^B$")
ax_B = temp.twiny()
ax_B.set_xlabel("$x_1^B$")
ax_B.invert_xaxis()
ax_B.invert_yaxis()
ax_B.set_xlim([1, 0])
ax_B.set_ylim([1, 0])

# Show the plot
plt.title('Edgeworth Box')
plt.legend()
plt.grid(True)
John = plt.show()


def plt(name):
    plt.show(name)
    return nameofplot



########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########
########## 2 ##########

p1values = np.linspace(0.5, 2.5, N+1)
eps1_values = []
eps2_values = []

for p1 in p1values:
    eps1, eps2 = economy.check_market_clearing(p1)
    eps1_values.append(eps1)
    eps2_values.append(eps2)

#print(eps1_values, eps2_values)
# Plot the excess demand functions
plt.figure(figsize=(10, 6))
plt.plot(p1values, eps1_values, label='eps1')
plt.plot(p1values, eps2_values, label='eps2')
plt.xlabel('p1 values')
plt.ylabel('Excess demand')
plt.title('Excess demand for different p1 values')
plt.legend()
plt.show()



### Spørg holdunderviser
    
########## 3 ##########
########## 3 ##########
########## 3 ##########
########## 3 ##########
########## 3 ##########
########## 3 ##########

def find_equilibrium(economy):
    # Define the excess demand function as a function of p1
    def excess_demand(p1):
        eps1, eps2 = economy.check_market_clearing(p1)
        return np.array([eps1, eps2])

    # Call the root finder
    result = minimize(lambda p1: np.sum(excess_demand(p1)**2), x0=1, method='Nelder-Mead', tol=1e-8)
    p1_star = result.x[0]
    return p1_star

print(find_equilibrium(economy))



########## 4 ##########
########## 4 ##########
########## 4 ##########
########## 4 ##########
########## 4 ##########
########## 4 ##########

def negative_utility_A(p1):
    # Get the demand for B given the price p1
    x1B, x2B = economy.demand_B(p1)
    
    # Calculate the remaining goods for A after B's consumption
    x1A_remaining = 1 - x1B
    x2A_remaining = 1 - x2B
    
    # The utility function for A expects positive consumption, if negative we return a large number
    if x1A_remaining < 0 or x2A_remaining < 0:
        return 1e6  # A large number to indicate a bad utility (not feasible)
    
    # Get the utility for A with the remaining goods
    utility_A = economy.utility_A(x1A_remaining, x2A_remaining)
    
    # We return the negative utility because we want to maximize the utility,
    # but the optimizer minimizes the function
    return -utility_A

# Find the price p1 that maximizes utility for A (minimizes the negative utility)
res = minimize_scalar(negative_utility_A, bounds=(0.5, 2.5), method='bounded')

# The optimal price p1
optimal_p1 = res.x
optimal_p1, -res.fun  # We negate the fun value to get the actual utility

print(optimal_p1, -res.fun)


# # a. total endowment
# # w1bar = 1.0
# # w2bar = 1.0

# # b. figure set up
# fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
# ax_A = fig.add_subplot(1, 1, 1)

# ax_A.set_xlabel("$x_1^A$")
# ax_A.set_ylabel("$x_2^A$")

# temp = ax_A.twinx()
# temp.set_ylabel("$x_2^B$")
# ax_B = temp.twiny()
# ax_B.set_xlabel("$x_1^B$")
# ax_B.invert_xaxis()
# ax_B.invert_yaxis()

# # A
# ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')

# # limits
# ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
# ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
# ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
# ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

# ax_A.set_xlim([-0.1, w1bar + 0.1])
# ax_A.set_ylim([-0.1, w2bar + 0.1])    
# ax_B.set_xlim([w1bar + 0.1, -0.1])
# ax_B.set_ylim([w2bar + 0.1, -0.1])

# ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0))


# ###################
# ###################
# ###################
# ###################
# ###################

# # x parameters

# alpha_val = 1/3
# beta_val = 2/3

# # x Utility functions

# def utility_A(x1A,x2A):
#     """ utility function for agent A
    
#     Args:
    
#         x1A (float): consumption of good 1 for agent A
#         x2A (float): consumption of good 2 for agent A
        
#     Returns:
    
#         uA (float): utility of agent A
    
#     """
    
#     uA = x1A**alpha_val * x2A**(1-alpha_val)
#     return uA

# utility_A(1,1)

# def utility_B(x1B,x2B):
#     """ utility function for agent B
    
#     Args:
    
#         x1B (float): consumption of good 1 for agent B
#         x2B (float): consumption of good 2 for agent B
        
#     Returns:
    
#         uB (float): utility of agent B
    
#     """
    
#     uB = x1B**beta_val * x2B**(1-beta_val)
#     return uB


# # x. demand functions
# def square(x):
#     """ square numpy array
    
#     Args:
    
#         x (ndarray): input array
        
#     Returns:
    
#         y (ndarray): output array
    
#     """
    
#     y = x**2
#     return y

# print("Hello World!")
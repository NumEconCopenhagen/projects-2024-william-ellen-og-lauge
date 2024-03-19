

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
# ax_A.plot(par.w1B, par.w2B, 'b*', markersize=10, label='Initial Endowment B')

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
import matplotlib.pyplot as plt

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



########## 4a ##########
########## 4a ##########
########## 4a ##########
########## 4a ##########
########## 4a ##########
########## 4a ##########

import numpy as np

utility_A_values = []

for p1 in p1values:
    x1A, x2A = economy.demand_A(p1)
    utility_A = economy.utility_A(x1A, x2A)
    utility_A_values.append(utility_A)
    max_utility_A = max(utility_A_values)
    max_utility_A_index = utility_A_values.index(max_utility_A)

print(max_utility_A, max_utility_A_index)

########## 4b ##########
########## 4b ##########
########## 4b ##########
########## 4b ##########




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
res = minimize_scalar(negative_utility_A, bounds=(0.00000, 15), method='bounded')

# The optimal price p1
optimal_p1 = res.x
optimal_p1, -res.fun  # We negate the fun value to get the actual utility

print(optimal_p1, -res.fun)




########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########

constant_utility_B = economy.utility_B(par.w1B, par.w2B)
# Define the constraint
cons = ({'type': 'eq', 'fun': lambda x:  economy.utility_B(x[0], x[1]) - constant_utility_B})

# Initial guess
x0 = np.array([0.5, 0.5])

# Define the objective function
def objective(x):
    return -economy.utility_A(x[0], x[1])  # We negate the utility to minimize

# Perform the optimization
res = minimize(objective, x0, constraints=cons, method='SLSQP')

# Print the result
print('Optimal allocation:', res.x)


initial_utility_B = economy.utility_B(par.w1B, par.w2B)

# Define the constraint for B's utility
def constraint_allocation(x):
    x1A, x2A = x
    x1B = 1 - x1A
    x2B = 1 - x2A
    return economy.utility_B(x1B, x2B) - initial_utility_B

# Define the optimization problem
def objective(x):
    x1A, x2A = x
    return -economy.utility_A(x1A, x2A)  # negative because minimize will be used

# Starting guess for A's allocation
x0 = [economy.par.w1A, economy.par.w2A]

# Define the constraints dictionary
cons = ({'type': 'ineq', 'fun': constraint_allocation})

# Run the optimizer
res = minimize(objective, x0, constraints=cons)

# Check if the optimization was successful
if res.success:
    allocated_x1A, allocated_x2A = res.x
    utility_A = -res.fun
    print(f"Allocation for A: x1 = {allocated_x1A}, x2 = {allocated_x2A}, with utility = {utility_A}")
else:
    print("Optimization was not successful.")


########## 5b ##########
########## 5b ##########
########## 5b ##########
########## 5b ##########
########## 5b ##########
    

########## 6a ##########
########## 6a ##########
########## 6a ##########
########## 6a ##########
########## 6a ##########

# Objective function for the social planner
def objective_function(x):
    x1A, x2A, x1B, x2B = x
    return -(economy.utility_A(x1A, x2A) + economy.utility_B(x1B, x2B))

# Constraints for the optimization problem
def constraint1(x):
    return x[0] + x[2] - economy.par.w1A - economy.par.w1B

def constraint2(x):
    return x[1] + x[3] - economy.par.w2A - economy.par.w2B

# Initial guess for consumption allocations
x0 = [0.5, 0.5, 0.5, 0.5]  

# Bounds for consumption allocations (non-negative consumption)
bounds = [(0, None)] * 4  # Bounds for each variable

# Defining the constraints
cons = [{'type': 'eq', 'fun': constraint1}, {'type': 'eq', 'fun': constraint2}]

# Solving the optimization problem
result = minimize(objective_function, x0, bounds=bounds, constraints=cons)

# Extracting the optimal allocation
optimal_allocation = result.x

print("Optimal allocation:")
print("x1A:", optimal_allocation[0])
print("x2A:", optimal_allocation[1])
print("x1B:", optimal_allocation[2])
print("x2B:", optimal_allocation[3])

########## 7 ##########
########## 7 ##########
########## 7 ##########
########## 7 ##########
########## 7 ##########

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

# Step 1: Generate set W
num_elements = 50
w1A_values = np.random.uniform(0, 1, num_elements)
w2A_values = np.random.uniform(0, 1, num_elements)

# Create a scatter plot for the Edgeworth box
fig = plt.figure(frameon=False, figsize=(8, 8), dpi=100)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Edgeworth Box with Market Equilibrium Allocations')

# Plot the endowments
plt.scatter(w1A_values, w2A_values, color='green', label='Endowments')
plt.scatter(economy.par.w1B, economy.par.w2B, color='black', label='Endowments')

# Iterate through each endowment and plot the corresponding allocations
for economy.par.w1A, economy.par.w2A in zip(w1A_values, w2A_values):
    # Define objective function to minimize excess demand
    def objective_function(p):
        p1, p2 = p
        x1A = economy.par.alpha * (p1 * economy.par.w1A + p2 * economy.par.w2A) / p1
        x2A = (1 - economy.par.alpha) * (p1 * economy.par.w1B + p2 * economy.par.w2B) / p2
        x1B = economy.par.beta * (p1 * economy.par.w1A + p2 * economy.par.w2A) / p1
        x2B = (1 - economy.par.beta) * (p1 * economy.par.w1B + p2 * economy.par.w2B) / p2
        excess1 = x1A + x1B - economy.par.w1A - economy.par.w1B
        excess2 = x2A + x2B - economy.par.w2A - economy.par.w2B
        return excess1**2 + excess2**2

    # Initial guess for prices
    p0 = [1.0, 1.0]

    # Minimize the objective function to find market clearing prices
    result = minimize(objective_function, p0)

    # Extract the market clearing prices
    p1, p2 = result.x

    # Calculate optimal allocations for individuals A and B
    x1A = economy.par.alpha * (p1 * economy.par.w1A + p2 * economy.par.w2A) / p1
    x2A = (1 - economy.par.alpha) * (p1 * economy.par.w1B + p2 * economy.par.w2B) / p2
    x1B = economy.par.beta * (p1 * economy.par.w1A + p2 * economy.par.w2A) / p1
    x2B = (1 - economy.par.beta) * (p1 * economy.par.w1B + p2 * economy.par.w2B) / p2

    # Plot the allocation in the Edgeworth box
    plt.scatter(x1A, x2A, color='blue', alpha=0.5)
    plt.scatter(x1B, x2B, color='red', alpha=0.5)

plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


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
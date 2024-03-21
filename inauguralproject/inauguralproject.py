

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

# Testing...
# results = ExchangeEconomyClass().utility_A(1,1)
# print(results)
    
# creating an instance of the class
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
# John = plt.show()


# def plt(name):
#     plt.show(name)
#     return nameofplot



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
# plt.show()




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

#One interpretation

utility_values = []
optimal_allocations = []
N = 75
p1values = np.linspace(0.5, 2.5, N+1)
for p1 in p1values:
    x1B, x2B = economy.demand_B(p1)
    # Check if the allocation is feasible (non-negative for both A and B)
    if x1B >= 0 and x2B >= 0 and (1 - x1B) >= 0 and (1 - x2B) >= 0:
        # Utility of A given her allocation (the rest of the endowment)
        utility_A = economy.utility_A(1 - x1B, 1 - x2B)
        utility_values.append(utility_A)
        optimal_allocations.append((1 - x1B, 1 - x2B))
    else:
        # Append a very low utility value for non-feasible allocations
        utility_values.append(-np.inf)
        optimal_allocations.append((None, None))

# Find the index of the maximum utility value excluding non-feasible allocations
max_utility_index = np.argmax(utility_values)  # This will ignore -np.inf values
max_utility = utility_values[max_utility_index]
optimal_allocation_A = optimal_allocations[max_utility_index]
optimal_p1 = p1values[max_utility_index]

# Print the results
print("4.a Index of max utility:", max_utility_index)
print("Max utility:", max_utility)
print("Optimal allocation for A:", optimal_allocation_A)
print("Optimal price p1:", optimal_p1)

#Another interpretation    

utility_A_values = []
optimal_x1A = None
optimal_x2A = None

for p1 in p1values:
    x1A, x2A = economy.demand_A(p1)
    
    # Clip the values to ensure they are between 0 and 1
    x1A = max(0, min(x1A, 1))
    x2A = max(0, min(x2A, 1))
    
    utility_A = economy.utility_A(x1A, x2A)
    utility_A_values.append(utility_A)
    if utility_A == max(utility_A_values):
        optimal_x1A = x1A
        optimal_x2A = x2A

max_utility_A = max(utility_A_values)

print("Optimal utility for A:", max_utility_A)
print("Optimal x1A value:", optimal_x1A)
print("Optimal x2A value:", optimal_x2A)


########## 4b ##########
########## 4b ##########
########## 4b ##########
########## 4b ##########

# Instantiate the ExchangeEconomyClass
economy = ExchangeEconomyClass()

def negative_utility_A(p1):
    # Get the demand for A given the price p1
    x1A, x2A = economy.demand_A(p1)
    
    # Clip the values to ensure they are between 0 and 1
    x1A = max(0, min(x1A, 1))
    x2A = max(0, min(x2A, 1))
    
    # Calculate the utility for A
    utility_A = economy.utility_A(x1A, x2A)
    
    # Return the negative utility because we want to maximize the utility,
    # but the optimizer minimizes the function
    return -utility_A

# Find the price p1 that maximizes utility for A (minimizes the negative utility)
res = minimize_scalar(negative_utility_A, bounds=(0.00000, 15), method='bounded')

# The optimal price p1
optimal_p1 = res.x

# Calculate the optimal allocation for consumer A given the optimal price p1
x1A_optimal, x2A_optimal = economy.demand_A(optimal_p1)

print("Optimal Price p1:", optimal_p1)
print("Optimal Allocation for Consumer A: x1A =", x1A_optimal, ", x2A =", x2A_optimal)

########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########
########## 5a ##########

# The `results` function has some issues that need to be fixed. Specifically:
# 1. In `economy.optimize_allocation()`, it returns only two values, but in the `results` function, 
#    there is an attempt to unpack four values from it.
# 2. The `final_utility_B` calculation seems to be using the wrong indices from `optimal_allocation_A`.

# Let's correct the `results` function and define a print command to print all the outputs.

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

# Let's call the function to print the results.
results(economy)


########## 5b ##########
########## 5b ##########
########## 5b ##########
########## 5b ##########
########## 5b ##########
# Define the utility of B at the endowment within the function
endowment_utility_B = economy.utility_B(economy.par.w1B, economy.par.w2B)

# Redefine the objective and constraints using the correct scope
def objective(x):
    return -economy.utility_A(x[0], x[1])

def constraint(x):
    return economy.utility_B(1-x[0], 1-x[1]) - endowment_utility_B

# Constraint dictionary
con = {'type': 'ineq', 'fun': constraint}

# Bounds for x1A and x2A (can't be negative or greater than total endowment)
bnds = ((0, 1), (0, 1))

# Initial guess (starting from A's endowment)
x0 = [economy.par.w1A, economy.par.w2A]

# Run the optimization again with the correct definition of endowment_utility_B in scope
solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=con)

# If the optimizer found a solution, extract it. Otherwise, set to None.
if solution.success:
    optimal_continuous_allocation_A = solution.x
    max_continuous_util_A = -solution.fun  # Negate because we minimized the negative utility
else:
    optimal_continuous_allocation_A = None
    max_continuous_util_A = None

optimal_continuous_allocation_A, max_continuous_util_A


def results5b(economy, optimal_allocation, max_utility):
    # Calculate initial utilities and endowments
    endowment_utility_A = economy.utility_A(economy.par.w1A, economy.par.w2A)
    endowment_utility_B = economy.utility_B(economy.par.w1B, economy.par.w2B)
    endow_A = (economy.par.w1A, economy.par.w2A)
    endow_B = (economy.par.w1B, economy.par.w2B)

    # Final utility and allocation for B based on A's optimal allocation
    final_allocation_B = (1 - optimal_allocation[0], 1 - optimal_allocation[1])
    final_utility_B = economy.utility_B(final_allocation_B[0], final_allocation_B[1])

    # Print results
    print(f"Initial utility for A: {endowment_utility_A}")
    print(f"Initial utility for B: {endowment_utility_B}")
    print(f"Initial endowment for A: {endow_A}")
    print(f"Initial endowment for B: {endow_B}")
    print(f"Optimal continuous allocation for A: {optimal_allocation}")
    print(f"Maximum continuous utility for A: {max_utility}")
    print(f"Final allocation for B: {final_allocation_B}")
    print(f"Final utility for B: {final_utility_B}")

# Call the results5b function with the results of the continuous optimization
results5b(economy, optimal_continuous_allocation_A, max_continuous_util_A)


    

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
# plt.show()


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
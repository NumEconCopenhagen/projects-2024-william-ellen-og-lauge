

# a. imports
from types import SimpleNamespace
import numpy as np
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

# access the par.w1A attribute
w1A = economy.par.w1A
w2A = economy.par.w2A
w1B = economy.par.w1B
w2B = economy.par.w2B
# use it in the utility_A function
UtilA_endowment = economy.utility_A(w1A, w2A)
UtilB_endowment = economy.utility_B(w1B, w2B)

# From chat

# Generate points in the Edgeworth box corresponding to feasible allocations
N = 75

x1_vals = np.linspace(0, 1, N+1)
x2_vals = np.linspace(0, 1, N+1)
feasible_allocations = []
for x1A in x1_vals:
    for x2A in x2_vals:
        # Calculate corresponding allocations for individual B
        x1B = 1 - x1A
        x2B = 1 - x2A
        # Check if allocation satisfies utility conditions
        if (economy.utility_A(x1A, x2A) >= economy.utility_A(economy.par.w1A, economy.par.w2A) and
            economy.utility_B(x1B, x2B) >= economy.utility_B(economy.par.w1B, economy.par.w2B)):
            feasible_allocations.append((x1A, x2A))

# Plot the Edgeworth box and the feasible allocation set
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [1, 0], 'k--')  # Line of perfect equality
plt.plot([0, w1A], [economy.par.w2A, economy.par.w2A], 'r--')  # Endowment for individual A
plt.plot([economy.par.w1B, 1], [economy.par.w2B, economy.par.w2B], 'b--')  # Endowment for individual B
plt.scatter(*zip(*feasible_allocations), color='g', marker='o', label='Feasible Allocations')
plt.xlabel('$x_1^A$')
plt.ylabel('$x_2^A$')
plt.title('Edgeworth Box with Feasible Allocations')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



# a. total endowment
# w1bar = 1.0
# w2bar = 1.0

# b. figure set up
fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
ax_A = fig.add_subplot(1, 1, 1)

ax_A.set_xlabel("$x_1^A$")
ax_A.set_ylabel("$x_2^A$")

temp = ax_A.twinx()
temp.set_ylabel("$x_2^B$")
ax_B = temp.twiny()
ax_B.set_xlabel("$x_1^B$")
ax_B.invert_xaxis()
ax_B.invert_yaxis()

# A
ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')

# limits
ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

ax_A.set_xlim([-0.1, w1bar + 0.1])
ax_A.set_ylim([-0.1, w2bar + 0.1])    
ax_B.set_xlim([w1bar + 0.1, -0.1])
ax_B.set_ylim([w2bar + 0.1, -0.1])

ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0))


###################
###################
###################
###################
###################

# x parameters

alpha_val = 1/3
beta_val = 2/3

# x Utility functions

def utility_A(x1A,x2A):
    """ utility function for agent A
    
    Args:
    
        x1A (float): consumption of good 1 for agent A
        x2A (float): consumption of good 2 for agent A
        
    Returns:
    
        uA (float): utility of agent A
    
    """
    
    uA = x1A**alpha_val * x2A**(1-alpha_val)
    return uA

utility_A(1,1)

def utility_B(x1B,x2B):
    """ utility function for agent B
    
    Args:
    
        x1B (float): consumption of good 1 for agent B
        x2B (float): consumption of good 2 for agent B
        
    Returns:
    
        uB (float): utility of agent B
    
    """
    
    uB = x1B**beta_val * x2B**(1-beta_val)
    return uB


# x. demand functions
def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y

print("Hello World!")
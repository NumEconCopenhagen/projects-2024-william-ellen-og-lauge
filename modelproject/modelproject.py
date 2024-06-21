import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from ipywidgets import interact, FloatSlider

# AS/AD model for a closed economy in the short run

class ASAD:
    def __init__(self, T, alpha=0.7, gamma=0.075, tol=0.01, z=0, s=0, z_duration=0, s_duration=0): 
        self.alpha = alpha 
        self.gamma = gamma 
        self.tol = tol
        self.T = T
        self.z = z
        self.s = s
        self.z_duration = z_duration
        self.s_duration = s_duration
        self.delta = 0.97
    
    def solve_model(self):
        self.yhat_vec = []
        self.pihat_vec = []
        self.social_loss_vec = []
        self.t_vec = []
        for t in range(self.T):
            yhat = pihat = z = s = 0
            if t <= self.z_duration:
                z = self.z
            if t <= self.s_duration:
                s = self.s
            if t > 0:
                yhat = (z - self.alpha * self.pihat_vec[t - 1] - self.alpha * s) / (1 + self.alpha * self.gamma)
                pihat = (self.pihat_vec[t - 1] + self.gamma * z + s) / (1 + self.alpha * self.gamma)
            self.yhat_vec.append(yhat)
            self.pihat_vec.append(pihat)
            # Calculate social loss as a simple function of the absolute values of gaps
            social_loss = 1000 * (abs(yhat) + abs(pihat))  # scale factor for visualization
            self.social_loss_vec.append(social_loss)
            self.t_vec.append(t)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Output gap and Inflation gap")
        plt.legend()
        plt.show()

    def plot_ad_as(self):
        
        """ Creating and plotting the AD and AS curves of the given output and inflation gaps
        args:
        self (class): class with initial values for the parameters
        
        Returns: 
        
        original_LRAD (list): list of original LRAD curve
        lras_curve (list): list of LRAS curve
        plot (plot): plot of AD and AS curves

        
        Returns: Temporary shocks:
        
        ad_curve_t (list): list of AD curves
        as_curve_t (list): list of AS curves
        
        Returns: Permenant shocks:
        
        adjusted_LRAD (list): list of adjusted LRAD curve
        LRAS2 (list): list of LRAS2 curve
        
        """
        # a. Create y_values & pi_hat
        
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pihat_vec
        # b. Create functions for AD and AS curves
        
        def ad_function(alpha, y, t, z, z_duration):
            if t <= z_duration:
                z_t = z
            else:
                z_t = 0
            return (-1/alpha)*(y-z_t)

        def as_function(alpha, pi_1, gamma, y, t, s, s_duration):
            
            if t <= s_duration:
                s_t = s
            else:
                s_t = 0
            return pi_1 + gamma * y + s_t

        # Initiate plot
        
        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve") #LRAS curve
        
        # The case for short-term shocks
        if self.s_duration < self.T:
            
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0') #LRAS curve
            for t in range(self.T): #plotting AD curve for each period
                
                ad_curve_t = ad_function(self.alpha, y_values, t, self.z, self.z_duration)
                plt.plot(y_values, ad_curve_t, color="blue")
            # Creating and plotting the SRAS curves
            
            for t in range(self.T):
                if t == 0:
                    pi_1 = 0
                else:
                    pi_1 = pi_hat[t-1]
                    
                as_curve_t = as_function(self.alpha, pi_1, self.gamma, y_values, t, self.s, self.s_duration)
                plt.plot(y_values, as_curve_t, color="red")

            # Original LRAD curve when s_duration < self.T
            
            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")
        # The case for permanent shocks
        
        if self.s_duration == self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            # Original LRAD curve
            
            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")

            # New LRAS (LRAS2)
            plt.axvline(x=-self.s, color="red", label="LRAS2 curve")

            # Adjusted LRAD2 curve (Central Bank response)
            
            if self.s < 0:  # Positive permanent supply shock
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
                
            else:
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
            plt.plot(y_values, adjusted_LRAD2, color="blue", linestyle="--", label="Adjusted LRAD2 curve")

        # Details for the plot
        
        plt.annotate(r'$\overline{y}$', xy=(-0.0015, -0.0125), fontsize=12)
        plt.annotate(r'$\overline{\pi}$', xy=(-0.0105, -0.0015), fontsize=12)
        plt.xlabel(r'Output gap $(\hat{y})$')
        plt.ylabel(r'Inflation gap $(\hat{\pi})$')
        
        if self.z_duration > 0 and self.z_duration < self.T:
            plt.title(f"Figure 2: {self.z_duration} period positive demand shock")
            
        elif self.z_duration == 0 and self.s_duration == self.T:
            plt.title(f"Figure 2: Permanent supply shock and Central Bank response")
        plt.grid()
        plt.show()

    def plot_social_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.social_loss_vec, label="Social Loss", color='red', linestyle='--')
        plt.xlabel("Periods")
        plt.ylabel("Social Loss")
        plt.title("Convergence of Social Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def social_loss_gamma(self, gamma, seed):
        np.random.seed(seed)
        # Simulating random shocks over T periods
        shocks = np.random.normal(0, 1, self.T)
        social_loss = 0
        pihat = 0
        for t in range(self.T):
            # Output response to shock based on current gamma
            yhat = shocks[t] - self.alpha * pihat
            pihat = pihat + gamma * yhat  # Adjust inflation based on gamma and output gap
            # Calculate social loss as sum of squares of output and inflation gaps
            social_loss += (yhat**2 + pihat**2)
        return social_loss / self.T  # Average loss over periods
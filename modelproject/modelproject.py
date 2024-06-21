import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

class ASAD:
    def __init__(self, T, alpha=0.7, gamma=1, tol=0.01, z=0, s=0, z_duration=0, s_duration=0): 
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
            social_loss =  (abs(yhat) + abs(pihat))  # scale factor for visualization
            self.social_loss_vec.append(social_loss)
            self.t_vec.append(t)

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_vec, self.yhat_vec, label="Output gap")
        plt.plot(self.t_vec, self.pihat_vec, label="Inflation gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Figure 1: Output gap and Inflation gap")
        plt.legend()
        plt.show()

    def plot_ad_as(self):
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pihat_vec

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

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve")
        
        if self.s_duration < self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            for t in range(self.T):
                ad_curve_t = ad_function(self.alpha, y_values, t, self.z, self.z_duration)
                plt.plot(y_values, ad_curve_t, color="blue")
            
            for t in range(self.T):
                if t == 0:
                    pi_1 = 0
                else:
                    pi_1 = pi_hat[t-1]
                as_curve_t = as_function(self.alpha, pi_1, self.gamma, y_values, t, self.s, self.s_duration)
                plt.plot(y_values, as_curve_t, color="red")

            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")

        if self.s_duration == self.T:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            original_LRAD = ad_function(self.alpha, y_values, 0, 0, self.z_duration)
            plt.plot(y_values, original_LRAD, color="blue", label="Original LRAD curve")
            plt.axvline(x=-self.s, color="red", label="LRAS2 curve")

            if self.s < 0: 
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
            else:
                adjusted_LRAD2 = ad_function(self.alpha, y_values, 0, -self.s, self.z_duration)
            plt.plot(y_values, adjusted_LRAD2, color="blue", linestyle="--", label="Adjusted LRAD2 curve")

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

    def optimize_alpha(self):
        def social_loss(alpha):
            self.alpha = alpha
            self.solve_model()
            return np.sum(self.social_loss_vec)
        
        initial_guess = [self.alpha]
        result = opt.minimize(social_loss, initial_guess, method='trust-constr')
        
        optimal_alpha = result.x[0]
        optimal_social_loss = result.fun
        
        return optimal_alpha, optimal_social_loss


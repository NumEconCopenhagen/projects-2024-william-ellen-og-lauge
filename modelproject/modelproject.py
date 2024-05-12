import sympy as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def ad(y, pi, pi_star, alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, tau, tau_bar, h, y_bar):
    z = alpha_1 / (1 + alpha_2 * b) * (g - g_bar) - alpha_3 / (1 + alpha_2 * b) * (tau - tau_bar)
    return y - y_bar - z + alpha * (pi - pi_star)

def as_curve(y, pi, pi_1, gamma, s, y_bar):
    return pi - pi_1 - gamma * (y - y_bar) - s

def plot_ad_as(s=0, y_bar=100, pi_star=2, alpha_1=1, alpha_2=1, alpha_3=1, b=1, g=1, g_bar=1, tau=1, tau_bar=1, h=1, gamma=0.5):
    # Create a range of output levels
    y_range = np.linspace(y_bar - 5, y_bar + 5, 100)

    # Derive the AD and AS curves
    alpha = alpha_2 * h / (1 + alpha_2 * b)
    ad_curve = ad(y_range, pi_star, pi_star, alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, tau, tau_bar, h, y_bar)
    as_curve_0 = as_curve(y_range, pi_star, pi_star, gamma, 0, y_bar)
    as_curve_1 = as_curve(y_range, pi_star, pi_star, gamma, s, y_bar)

    # Plotting
    plt.plot(y_range, ad_curve, label='AD')
    plt.plot(y_range, as_curve_0, label='AS (s=0)', color='black')
    plt.plot(y_range, as_curve_1, label='AS (s={:.2f})'.format(s), color='red')

    plt.xlabel('Output')
    plt.ylabel('Inflation (in percentage points)')
    plt.legend()



##TODO IMPORT FROM OTHER STUDENT


class AS_AD_model:
    def __init__(self):
        #Defining the symbols using sympy
        self.y, self.pi, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s, self.y_bar, self.pi_star, self.tau, self.tau_bar = sm.symbols(
            'y pi g b alpha_1 alpha_2 alpha_3 h s y_bar pi_star tau tau_bar')

        #Defining the AD-curve
        z = self.alpha_1 / (1 + self.alpha_2 * self.b) * (self.g - self.y_bar)- self.alpha_3 / (1 + self.alpha_2 * self.b) * (self.tau - self.tau_bar)
        self.alpha = self.alpha_2 * self.h / (1 + self.alpha_2 * self.b)
        self.AD = self.y - self.y_bar + self.alpha * (self.pi - self.pi_star) - z 
### TODO Remember that we changed the sign of +self.alpha

        #Defining the AS-curve
        self.gamma = sm.symbols('gamma')
        self.AS = self.pi - self.pi_star - self.gamma * (self.y - self.y_bar) - self.s

        #Steady state equation
        self.steady_state_eq = sm.solve(sm.Eq(self.AD, self.AS), self.y)[0]

        #Using lambdify to define the SS function
        self.ss_func = sm.lambdify(
            (self.pi, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s, self.y_bar, self.pi_star, self.gamma, self.tau, self.tau_bar),
            self.steady_state_eq)
        

    def ad(self, y, pi, pi_star, alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, h, tau, tau_bar):
        #Defining the AD-curve equation
        z = alpha_1 / (1 + alpha_2 * b) * (g - g_bar) - alpha_3 / (1 + alpha_2 * b) * (tau - tau_bar)

        return y - self.y_bar - z + alpha * (pi - pi_star)

    def as_curve(self, y, pi, pi_1, gamma, s):
        #Defining the AS-curve equation
        return pi - pi_1 - gamma * (y - self.y_bar) - s

    def analyze_policy_intervention(self, interest_rate):
        #Defining the parameter values
        self.y_bar = 100
        self.pi_star = 2
        self.alpha_1 = 1
        self.alpha_2 = 1
        self.alpha_3 = 1
        self.b = 1
        self.g = 1
        self.g_bar = 1
        self.h = 1
        self.alpha = self.alpha_2 * self.h / (1 + self.alpha_2 * self.b)
        self.gamma = 0.5
        self.tau = 1/2
        self.tau_bar = 1/2


        #Assuming no supply shocks 
        self.s = 0

        #Calculating steady state before policy intervention
        steady_state_before = self.ss_func(self.pi_star, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s,
                                           self.y_bar, self.pi_star, self.gamma, self.tau, self.tau_bar)

        #Applying the policy intervention (decrease in interest rate)
        self.s = interest_rate

        #Calculating steady state after policy intervention
        steady_state_after = self.ss_func(self.pi_star, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s,
                                           self.y_bar, self.pi_star, self.gamma, self.tau, self.tau_bar)

        return steady_state_before, steady_state_after
    

    def plot_ad_as(self, s=0):
        # Create a range of output levels
        y_range = np.linspace(self.y_bar - 5, self.y_bar + 5, 100)

        # Derive the AD and AS curves
        ad_curve = self.ad(y_range, self.pi_star, self.pi_star, self.alpha, self.alpha_1, self.alpha_2, self.alpha_3, self.b, self.g, self.g_bar, self.h, self.tau, self.tau_bar)
        as_curve_0 = self.as_curve(y_range, self.pi_star, self.pi_star, self.gamma, 0)
        as_curve_1 = self.as_curve(y_range, self.pi_star, self.pi_star, self.gamma, s)

        # Plotting
        plt.plot(y_range, ad_curve, label='AD')
        plt.plot(y_range, as_curve_0, label='AS (s=0)', color='black')
        plt.plot(y_range, as_curve_1, label='AS (s={:.2f})'.format(s), color='red')

        plt.xlabel('Output')
        plt.ylabel('Inflation (in percentage points)')
        plt.legend()
        plt.show()


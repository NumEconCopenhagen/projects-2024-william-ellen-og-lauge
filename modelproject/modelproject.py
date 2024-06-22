import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

class EconomicModel:
    def __init__(self, periods, alpha=0.7, gamma=0.075, tolerance=0.01, demand_shock=0.1, supply_shock=0.05, demand_duration=1, supply_duration=1): 
        # Initialize the parameters of our economic model
        self.alpha = alpha  # We set the responsiveness of inflation to the output gap
        self.gamma = gamma  # We determine the responsiveness of inflation to demand and supply shocks
        self.tolerance = tolerance  # We define our tolerance level for convergence
        self.periods = periods  # We specify the number of periods to simulate
        self.demand_shock = demand_shock  # We set the magnitude of the demand shock
        self.supply_shock = supply_shock  # We set the magnitude of the supply shock
        self.demand_duration = demand_duration  # We define the duration of the demand shock
        self.supply_duration = supply_duration  # We define the duration of the supply shock
        self.discount_factor = 0.97  # We apply a discount factor for future periods
    
    def calculate_model(self):
        # Next, we calculate the output and inflation gaps over time based on our shocks and model parameters
        self.output_gap = []  # We store the output gap over time
        self.inflation_gap = []  # We store the inflation gap over time
        self.loss_vector = []  # We keep track of the social loss over time
        self.time_vector = []  # We record the time periods
        
        for time in range(self.periods):
            y_gap = pi_gap = d_shock = s_shock = 0
            
            # We apply the demand shock if within the shock duration
            if time <= self.demand_duration:
                d_shock = self.demand_shock
            
            # We apply the supply shock if within the shock duration
            if time <= self.supply_duration:
                s_shock = self.supply_shock
            
            # Next, we calculate the output based on our shocks and the previous inflation gap
            if time > 0:
                y_gap = (d_shock - self.alpha * self.inflation_gap[time - 1] - self.alpha * s_shock) / (1 + self.alpha * self.gamma)
                pi_gap = (self.inflation_gap[time - 1] + self.gamma * d_shock + s_shock) / (1 + self.alpha * self.gamma)
            
            # We append the calculated gaps to their respective lists
            self.output_gap.append(y_gap)
            self.inflation_gap.append(pi_gap)
            
            # We calculate the social loss as a weighted sum of the absolute value of gaps
            loss = 1000 * (abs(y_gap) + abs(pi_gap))  # scaling for visualization
            self.loss_vector.append(loss)
            self.time_vector.append(time)

    def display_results(self):
        # We display a plot of the output and inflation gaps over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_vector, self.output_gap, label="Output Gap")
        plt.plot(self.time_vector, self.inflation_gap, label="Inflation Gap")
        plt.xlabel("Periods")
        plt.ylabel("Gap")
        plt.title("Output and Inflation Gaps Over Time")
        plt.legend()
        plt.show()

    def display_ad_as(self):
        # We visualize the Aggregate Demand-Aggregate Supply (AD-AS) curves
        y_vals = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.inflation_gap

        def ad_curve(alpha, y, time, d_shock, demand_duration):
            # We define the Aggregate Demand (AD) curve equation
            d_shock_t = d_shock if time <= demand_duration else 0
            return (-1 / alpha) * (y - d_shock_t)

        def as_curve(alpha, prev_pi, gamma, y, time, s_shock, supply_duration):
            # We define the Aggregate Supply (AS) curve equation
            s_shock_t = s_shock if time <= supply_duration else 0
            return prev_pi + gamma * y + s_shock_t

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="Long Run AS Curve")
        
        if self.supply_duration < self.periods:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            for time in range(self.periods):
                ad_t = ad_curve(self.alpha, y_vals, time, self.demand_shock, self.demand_duration)
                plt.plot(y_vals, ad_t, color="blue")
            
            for time in range(self.periods):
                prev_pi = pi_hat[time-1] if time > 0 else 0
                as_t = as_curve(self.alpha, prev_pi, self.gamma, y_vals, time, self.supply_shock, self.supply_duration)
                plt.plot(y_vals, as_t, color="red")

            original_lrad = ad_curve(self.alpha, y_vals, 0, 0, self.demand_duration)
            plt.plot(y_vals, original_lrad, color="blue", label="Original LRAD Curve")

        if self.supply_duration == self.periods:
            plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')
            original_lrad = ad_curve(self.alpha, y_vals, 0, 0, self.demand_duration)
            plt.plot(y_vals, original_lrad, color="blue", label="Original LRAD Curve")
            plt.axvline(x=-self.supply_shock, color="red", label="Adjusted LRAS Curve")

            adjusted_lrad2 = ad_curve(self.alpha, y_vals, 0, -self.supply_shock, self.demand_duration)
            plt.plot(y_vals, adjusted_lrad2, color="blue", linestyle="--", label="Adjusted LRAD Curve")

        plt.annotate(r'$\overline{y}$', xy=(-0.0015, -0.0125), fontsize=12)
        plt.annotate(r'$\overline{\pi}$', xy=(-0.0105, -0.0015), fontsize=12)
        plt.xlabel(r'Output Gap $(\hat{y})$')
        plt.ylabel(r'Inflation Gap $(\hat{\pi})$')
        
        # We add a title based on shock conditions
        if 0 < self.demand_duration < self.periods:
            plt.title(f"Positive Demand Shock for {self.demand_duration} Periods")
        elif self.demand_duration == 0 and self.supply_duration == self.periods:
            plt.title(f"Permanent Supply Shock with Central Bank Response")
        plt.grid()
        plt.show()

    def display_social_loss(self):
        # We display the plot of social loss over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_vector, self.loss_vector, label="Social Loss", color='red', linestyle='--')
        plt.xlabel("Periods")
        plt.ylabel("Social Loss")
        plt.title("Convergence of Social Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def find_optimal_parameters(self):
        # We find the optimal value of alpha that minimizes social loss
        def loss_function(params):
            alpha = params[0]
            self.alpha = alpha
            self.calculate_model()
            return np.sum(self.loss_vector)
        
        initial_guess = [self.alpha]
        bounds = [(0.1, 1.5)]  # We set bounds for alpha
        result = opt.minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        optimal_alpha = result.x[0]
        optimal_loss = result.fun
        
        return optimal_alpha, optimal_loss

    def output_gap_after_shock(self):
        # We calculate the absolute value of output gap after a shock
        self.calculate_model()
        if len(self.output_gap) > 1:
            return abs(self.output_gap[1])
        return float('inf')  # We return a large number if the output gap cannot be calculated

    def optimize_output_gap(self):
        # We optimize alpha to minimize output gap after a shock
        def gap_function(params):
            alpha = params[0]
            self.alpha = alpha
            self.calculate_model()
            output_gap = self.output_gap_after_shock()
            return output_gap
        
        initial_guess = [self.alpha]
        bounds = [(0.1, 25)]  # We set bounds for alpha
        result = opt.minimize(gap_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        optimal_alpha = result.x[0]
        optimal_output_gap = self.output_gap_after_shock()
        
        return optimal_alpha, optimal_output_gap


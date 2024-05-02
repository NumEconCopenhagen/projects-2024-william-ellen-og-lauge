import matplotlib.pyplot as plt
import numpy as np

class ASAD_OpenEconomy:
    # Class for the AS-AD model in an open economy with a fixed exchange rate
    def __init__(self, T, foreign_interest_rate, beta1=0.5, beta2=0.5, beta3=0.5, alpha=0.6, gamma=0.07, tol=0.01):
        self.T = T  # Number of periods
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.alpha = alpha  # Sensitivity of output to the output gap
        self.gamma = gamma  # Sensitivity of inflation to the output gap
        self.tol = tol  # Tolerance for convergence
        self.foreign_interest_rate = foreign_interest_rate  # Fixed foreign interest rate
        self.r = foreign_interest_rate  # Domestic interest rate initially set to foreign interest rate
        self.e = 0  # Initial expected exchange rate (no devaluation expected)
        self.pi = 0  # Initial inflation rate
        self.y = 0  # Initial output
        
    def solve_model(self):
        self.y_vec = [self.y]
        self.pi_vec = [self.pi]
        self.r_vec = [self.r]
        self.e_vec = [self.e]

        for t in range(1, self.T):
            # Expected inflation and output evolve over time
            expected_pi = self.pi_vec[-1]  # Inflation expectations are adaptive
            expected_y = self.y_vec[-1]    # Output expectations are adaptive
            
            # Calculate the output gap and inflation gap
            output_gap = self.beta1 * (expected_y + expected_pi - self.pi_vec[-1]) - self.beta2 * (self.r - expected_pi)
            inflation_gap = self.gamma * output_gap + np.random.normal(0, self.tol)  # Adding a shock term for variability
            
            # Update values
            self.y_vec.append(output_gap + self.y)
            self.pi_vec.append(inflation_gap + self.pi)
            self.r_vec.append(self.foreign_interest_rate)  # Domestic interest rate tracks the foreign rate
            self.e_vec.append(self.e)  # Exchange rate is fixed, no change
            
        self.t_vec = list(range(self.T))
    
    def plot_results(self):
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 2, 1)
        plt.plot(self.t_vec, self.y_vec, label='Output')
        plt.title('Output over Time')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.t_vec, self.pi_vec, label='Inflation')
        plt.title('Inflation over Time')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.t_vec, self.r_vec, label='Interest Rate')
        plt.title('Interest Rate')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.t_vec, self.e_vec, label='Exchange Rate')
        plt.title('Exchange Rate')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_ad_as(self):
        y_values = np.linspace(-0.01, 0.01, 100)
        pi_hat = self.pi_vec
        
        def ad_function(alpha, y, pi, t, r):
            return (-1/alpha)*(y - (r - pi))

        def as_function(gamma, y, pi_1, s):
            return pi_1 + gamma * y + s

        plt.figure(figsize=(10, 6))
        plt.axvline(x=0, color="red", label="LRAS curve")  # LRAS curve is vertical at the natural level of output

        for t in range(self.T):  # Plotting AD and AS curves for each period
            pi_1 = pi_hat[t - 1] if t > 0 else 0
            ad_curve = ad_function(self.alpha, y_values, pi_1, t, self.foreign_interest_rate)
            as_curve = as_function(self.gamma, y_values, pi_1, 0)  # No supply shocks are modeled in the simple version

            plt.plot(y_values, ad_curve, color="blue", alpha=0.5)  # Lower alpha to show multiple lines
            plt.plot(y_values, as_curve, color="red", alpha=0.5)

        # Assuming equilibrium y and pi are both zero (simplified)
        plt.axhline(y=0, color='black', linestyle=':', label='Long Run Inflation Gap = 0')

        plt.xlabel(r'Output gap $(\hat{y})$')
        plt.ylabel(r'Inflation gap $(\hat{\pi})$')
        plt.title("AD-AS Diagram for Open Economy with Fixed Exchange Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def solve_stochastic_shock(self, seed):
        """
        Solves the model with stochastic shocks to demand (z) and supply (s) using autoregressive processes.
        
        Args:
            seed (int): Seed for random number generator to ensure reproducibility.
        
        Updates:
            self.yhat_vec_stoc (list): List of output gaps under stochastic conditions.
            self.pihat_vec_stoc (list): List of inflation gaps under stochastic conditions.
            self.t_vec (list): List of periods.
        """
        np.random.seed(seed)  # Set the seed for reproducibility
        
        # Initialize shock vectors and response vectors
        self.z_vector = [0]  # Initial demand shock is zero
        self.s_vector = [0]  # Initial supply shock is zero
        self.yhat_vec_stoc = [0]  # Initial output gap is zero
        self.pihat_vec_stoc = [0]  # Initial inflation gap is zero
        self.t_vec = list(range(self.T))  # Time vector

        # Parameters for AR processes
        AR_s = 0.15  # Autoregression coefficient for supply shocks
        AR_z = 0.8   # Autoregression coefficient for demand shocks
        sigma_s = 0.2  # Standard deviation of supply shocks
        sigma_z = 1    # Standard deviation of demand shocks

        # Generate shocks and solve the model for each period
        for t in range(1, self.T):
            # Generate new shocks based on previous shocks
            new_s = self.s_vector[t-1] * AR_s + np.random.normal(0, sigma_s)
            new_z = self.z_vector[t-1] * AR_z + np.random.normal(0, sigma_z)
            self.s_vector.append(new_s)
            self.z_vector.append(new_z)

            # Calculate the output and inflation gaps
            yhat = (new_z - self.alpha * self.pihat_vec_stoc[t - 1] - self.alpha * new_s) / (1 + self.alpha * self.gamma)
            pihat = (self.pihat_vec_stoc[t - 1] + self.gamma * yhat)  # Simplified version

            # Update the vectors with new calculated values
            self.yhat_vec_stoc.append(yhat)
            self.pihat_vec_stoc.append(pihat)

# Create an instance of the model

model = ASAD_OpenEconomy(T=50, foreign_interest_rate=0.05, alpha=0.6, gamma=0.07)
model.solve_model()
model.plot_ad_as()
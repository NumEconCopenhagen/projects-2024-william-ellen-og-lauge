import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class AS_AD_model:
    def __init__(self):
        self.y, self.pi, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s, self.y_bar, self.pi_star, self.tau, self.tau_bar = sm.symbols(
            'y pi g b alpha_1 alpha_2 alpha_3 h s y_bar pi_star tau tau_bar')

        # Define the AD and AS curves using sympy expressions
        z = self.alpha_1 / (1 + self.alpha_2 * self.b) * (self.g - self.y_bar) - self.alpha_3 / (1 + self.alpha_2 * self.b) * (self.tau - self.tau_bar)
        self.alpha = self.alpha_2 * self.h / (1 + self.alpha_2 * self.b)
        self.AD = self.y - self.y_bar - z - self.alpha * (self.pi - self.pi_star)
        self.gamma = sm.symbols('gamma')
        self.AS = self.pi - self.pi_star - self.gamma * (self.y - self.y_bar) - self.s

        # Solve the steady state equation
        self.steady_state_eq = sm.solve(sm.Eq(self.AD, self.AS), self.y)[0]
        self.ss_func = sm.lambdify(
            (self.pi, self.g, self.b, self.alpha_1, self.alpha_2, self.alpha_3, self.h, self.s, self.y_bar, self.pi_star, self.gamma, self.tau, self.tau_bar),
            self.steady_state_eq)

        # Set parameters
        self.par = SimpleNamespace(y_bar=100, pi_star=2, alpha_1=1, alpha_2=1, alpha_3=1, b=1, g=1, g_bar=1, h=1, gamma=0.5, tau=0.5, tau_bar=0.5, s=0)

    def ad(self, y, y_bar, pi, pi_star, alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, tau, tau_bar, h):
        z = alpha_1 / (1 + alpha_2 * b) * (g - g_bar) - alpha_3 / (1 + alpha_2 * b) * (tau - tau_bar)
        return y - y_bar - z - alpha * (pi - pi_star)

    def as_curve(self, y, pi, pi_star, gamma, s, y_bar):
        return pi - pi_star - gamma * (y - y_bar) - s

    def analyze_policy_intervention(self, interest_rate):
        self.s = interest_rate

        steady_state_before = self.ss_func(self.par.pi_star, self.par.g, self.par.b, self.par.alpha_1, self.par.alpha_2, self.par.alpha_3, self.par.h, 0,
                                           self.par.y_bar, self.par.pi_star, self.par.gamma, self.par.tau, self.par.tau_bar)

        steady_state_after = self.ss_func(self.par.pi_star, self.par.g, self.par.b, self.par.alpha_1, self.par.alpha_2, self.par.alpha_3, self.par.h, self.s,
                                          self.par.y_bar, self.par.pi_star, self.par.gamma, self.par.tau, self.par.tau_bar)

        return steady_state_before, steady_state_after

def plot_supply_shock(s=0, y_bar=100, pi_star=2, alpha_1=1, alpha_2=1, alpha_3=1, b=1, g=1, g_bar=1, tau=1/2, tau_bar=1/2, h=1, gamma=0.5):
    y_range = np.linspace(y_bar - 5, y_bar + 5, 100)

    ad_curve = model.ad(y_range, y_bar, pi_star, pi_star, model.alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, tau, tau_bar, h)
    as_curve_0 = model.as_curve(y_range, pi_star, pi_star, gamma, 0, y_bar)
    as_curve_1 = model.as_curve(y_range, pi_star, pi_star, gamma, s, y_bar)

    plt.plot(y_range, ad_curve, label='AD', color='blue')  # AD curve in blue
    plt.plot(y_range, as_curve_0, label='AS (s=0)', color='red')  # AS curve without shock in red
    plt.plot(y_range, as_curve_1, label='AS (s={:.2f})'.format(s), color='green')  # AS curve with shock in green
    plt.ylim([-4, 4])

    plt.xlabel('Output')
    plt.ylabel('Inflation (in percentage points)')
    plt.legend()

def plot_demand_shock(s=0, y_bar=100, pi_star=2, alpha_1=1, alpha_2=1, alpha_3=1, b=1, g=1, g_bar=1, tau=1/2, tau_bar=1/2, h=1, gamma=0.5):
    y_range = np.linspace(y_bar - 5, y_bar + 5, 100)

    ad_curve_0 = model.ad(y_range, y_bar, pi_star, pi_star, 1, 1, 1, 1, 1, 1, 1, 1/2, 1/2, h)
    ad_curve_1 = model.ad(y_range, y_bar, pi_star, pi_star, model.alpha, alpha_1, alpha_2, alpha_3, b, g, g_bar, tau, tau_bar, h)
    as_curve_res = model.as_curve(y_range, pi_star, pi_star, gamma, s, y_bar)

    plt.plot(y_range, ad_curve_0, label='AD (z=0)', color='blue')  # AD curve without shock in blue
    plt.plot(y_range, ad_curve_1, label='AD', color='orange')  # AD curve with shock in orange
    plt.plot(y_range, as_curve_res, label='AS', color='red')  # AS curve in red
    plt.ylim([-4, 4])

    plt.xlabel('Output')
    plt.ylabel('Inflation (in percentage points)')
    plt.legend()

# Instantiate the AS_AD_model class
model = AS_AD_model()

# Example usage
plot_supply_shock(s=-0.5)
plt.title('Supply Shock (AS increases)')
plt.show()

plot_demand_shock(s=-0.5)
plt.title('Demand Shock (AD decreases)')
plt.show()

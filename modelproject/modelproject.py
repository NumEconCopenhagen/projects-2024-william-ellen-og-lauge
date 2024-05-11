import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def ad(y, pi, pi_star, alpha, alpha1, alpha2, alpha3, b, g, g_bar, tau, tau_bar, h, y_bar):
    z = alpha1 / (1 + alpha2 * b) * (g - g_bar) - alpha3 / (1 + alpha2 * b) * (tau - tau_bar)
    return y - y_bar - z + alpha * (pi - pi_star)

def as_curve(y, pi, pi_1, gamma, s, y_bar):
    return pi - pi_1 - gamma * (y - y_bar) - s

def plot_ad_as(s=0, tau=0, g=0, y_bar=100, pi_star=2, alpha1=1, alpha2=1, alpha3=1, b=1, tau_bar=1, h=1, gamma=0.5):
    # Create a range of output levels
    y_range = np.linspace(y_bar - 5, y_bar + 5, 100)

    # Derive the AD and AS curves
    alpha = alpha2 * h / (1 + alpha2 * b)
    ad_curve = ad(y_range, pi_star, pi_star, alpha, alpha1, alpha2, alpha3, b, g, g, tau, tau_bar, h, y_bar)
    as_curve_s = as_curve(y_range, pi_star, pi_star, gamma, s, y_bar)
    as_curve_tau = as_curve(y_range, pi_star, pi_star, gamma, tau, y_bar)
    as_curve_g = as_curve(y_range, pi_star, pi_star, gamma, g, y_bar)

    # Plotting
    plt.plot(y_range, ad_curve, label='AD')
    if s != 0:
        plt.plot(y_range, as_curve_s, label='AS (s={:.2f})'.format(s), color='red')
    plt.plot(y_range, as_curve_tau, label='AS (tau={:.2f})'.format(tau), color='blue')
    plt.plot(y_range, as_curve_g, label='AS (g={:.2f})'.format(g), color='green')

    plt.xlabel('Output')
    plt.ylabel('Inflation (in percentage points)')
    plt.legend()

# Interactive plot with FloatSliders for s, tau, and g
def plot_ad_as_interactive():
    interact(plot_ad_as, s=FloatSlider(min=-2, max=2, step=0.1, value=0, description='s'),
                        tau=FloatSlider(min=-2, max=2, step=0.1, value=0, description='tau'),
                        g=FloatSlider(min=-2, max=2, step=0.1, value=0, description='g'))

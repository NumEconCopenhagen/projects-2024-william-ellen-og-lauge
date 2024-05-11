import sympy as sm
import numpy as np

def ad(y, pi, pi_star, alpha, alpha1, alpha2, alpha3, b, g, g_bar, tau, tau_bar, h, y_bar):
    z = alpha1 / (1 + alpha2 * b) * (g - g_bar) - alpha3 / (1 + alpha2 * b) * (tau - tau_bar)
    return y - y_bar - z + alpha * (pi - pi_star)

def as_curve(y, pi, pi_1, gamma, s, y_bar):
    return pi - pi_1 - gamma * (y - y_bar) - s

def plot_ad_as(s, y_bar, pi_star, alpha, alpha1, alpha2, alpha3, b, g, g_bar, tau, tau_bar, h, gamma):
    # Create a range of output levels
    y_range = np.linspace(y_bar - 5, y_bar + 5, 100)

    # Derive the AD and AS curves
    ad_curve = ad(y_range, pi_star, pi_star, alpha, alpha1, alpha2, alpha3, b, g, g_bar, tau, tau_bar, h, y_bar)
    as_curve_0 = as_curve(y_range, pi_star, pi_star, gamma, 0, y_bar)
    as_curve_1 = as_curve(y_range, pi_star, pi_star, gamma, s, y_bar)

    # Plotting
    import matplotlib.pyplot as plt
    plt.plot(y_range, ad_curve, label='AD')
    plt.plot(y_range, as_curve_0, label='AS (s=0)', color='black')
    plt.plot(y_range, as_curve_1, label='AS (s={:.2f})'.format(s), color='red')

    plt.xlabel('Output')
    plt.ylabel('Inflation (in percentage points)')
    plt.legend()
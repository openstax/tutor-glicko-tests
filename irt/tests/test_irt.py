from irt_3pl import IRT3PL
import pandas as pd
import numpy as np
from scipy.stats import kendalltau


def test_irt_computation_1pl():
    # Generate some synthetic data and run things.  Verify that kendall tau is reasonable
    irt = IRT3PL(model_name='1pl')
    data = irt.generate_synthetic_data(100, 100, 1)
    irt.fit(data[0])
    theta_hat = irt.theta_df['theta'].values
    theta = data[2]['theta'].values
    gamma_hat = irt.gamma_df['gamma'].values
    alpha_hat = irt.alpha_df['alpha'].values
    k = kendalltau(theta, theta_hat)[0]
    assert np.mean((gamma_hat-0)**2) < 1e-3
    assert np.mean((alpha_hat - 1) ** 2) < 1e-3
    assert(k > 0.8)

def test_irt_computation_2pl():
    # Generate some synthetic data and run things.  Verify that kendall tau is reasonable
    irt = IRT3PL(model_name='2pl')
    data = irt.generate_synthetic_data(100, 100, 1)
    irt.fit(data[0])
    theta_hat = irt.theta_df['theta'].values
    theta = data[2]['theta'].values
    gamma_hat = irt.gamma_df['gamma'].values
    k = kendalltau(theta, theta_hat)[0]
    assert np.mean((gamma_hat-0)**2) < 1e-3
    assert(k > 0.8)

def test_irt_computation_3pl():
    # Generate some synthetic data and run things.  Verify that kendall tau is reasonable
    irt = IRT3PL(model_name='1pl')
    data = irt.generate_synthetic_data(100, 100, 1)
    irt.fit(data[0])
    theta_hat = irt.theta_df['theta'].values
    theta = data[2]['theta'].values
    k = kendalltau(theta, theta_hat)[0]
    assert(k > 0.8)

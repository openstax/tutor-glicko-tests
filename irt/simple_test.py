from irt_3pl import IRT3PL
import pandas as pd
import numpy as np
from scipy.stats import kendalltau


irt = IRT3PL(model_name='1pl')
Y_df, Y_full_df, theta_df, beta_df, alpha_df, gamma_df = irt.generate_synthetic_data(100, 90, .5)
theta = theta_df['theta'].values
theta.shape = (len(theta), 1)

beta = beta_df['beta'].values
beta.shape = (len(beta), 1)

alpha = alpha_df['alpha'].values
alpha.shape = (len(alpha), 1)

gamma = gamma_df['gamma'].values
gamma.shape = (len(gamma), 1)

irt.fit(Y_df)
pred_df_in = Y_df.sample(frac=.1)
pred_df_out = irt.predict(pred_df_in)
pred_df_out = pred_df_out.merge(pred_df_in)

theta_df = theta_df.merge(irt.theta_df, on='student')
kendalltau(theta_df['theta_x'], theta_df['theta_y'])

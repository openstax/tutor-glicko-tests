from irt_3pl import IRT3PL
import pandas as pd
import numpy as np
from scipy.stats import kendalltau


irt = IRT3PL()
stuff = irt.generate_synthetic_data(100, 90, .5)
Y_df = stuff[0]
irt.fit(Y_df)
pred_df_in = Y_df.sample(frac=.1)
pred_df_out = irt.predict(pred_df_in)
pred_df_out = pred_df_out.merge(pred_df_in)


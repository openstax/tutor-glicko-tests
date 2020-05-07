from glicko import Glicko2
import pandas as pd

N_students = 100
N_questions = 99
p_obs = .5

glicko = Glicko2()

# df_sample, df_full, df_students, df_questions = glicko.generate_synthetic_data(N_students, N_questions, p_obs)

# Run fit
glicko.df_entities = pd.DataFrame(
    {
        'entity': ['P1', 'P2', 'P3', 'P4'],
        'rating': [1500, 1400, 1550, 1700],
        'rd': [200, 30, 100, 300],
        'volatility': [.06, .06, .06, .06],
        'mu': [(r-1500)/glicko.default_standardization for r in [1500, 1400, 1550, 1700]],
        'phi': [rd/glicko.default_standardization for rd in [200, 30, 100, 300]]
    }
)
df_sample = pd.DataFrame(
    {
        'student': ['P1', 'P1', 'P1'],
        'question': ['P2', 'P3', 'P4'],
        'score': [1, 0, 0]
    }
)
glicko.fit(df_sample[['student', 'question', 'score']])
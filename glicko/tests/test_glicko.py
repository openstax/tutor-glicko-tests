from glicko import Glicko2
import pandas as pd
import numpy as np
from scipy.stats import kendalltau


def test_glicko_computation():
    # Numerical example taken from the original paper
    # See: http: // www.glicko.net / glicko / glicko2.pdf
    glicko = Glicko2()
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
    assert np.round(glicko.df_entities.iloc[0]['rating']) == 1464

def test_glicko_consistency():
    # Generate random data and then fit to data.  See if the rating order strongly matches the expectation
    np.random.seed(42)
    glicko = Glicko2()
    df_sample, df_full, df_students, df_questions = glicko.generate_synthetic_data(100, 100, 1)

    glicko.fit(df_sample[['student', 'question', 'score']])

    output = glicko.df_entities
    df_students = df_students[['entity_1', 'rating_1']].rename(columns={'entity_1': 'entity'})
    df_students = df_students.merge(output[['entity', 'rating']])
    kt = kendalltau(df_students['rating'], df_students['rating_1'])
    assert kt[0] > 0.85 # This indicates very strong agreement!

def test_prediction():
    # Generate random data and then fit to data.  Make sure we can do sensible predictions
    np.random.seed(42)
    glicko = Glicko2()
    df_sample, df_full, df_students, df_questions = glicko.generate_synthetic_data(100, 100, 0.5)

    # Compute the holdout data (in df_full, not in df_sample)
    df_holdout = df_full[~df_full.index.isin(df_sample.index.values)]

    glicko.fit(df_sample[['student', 'question', 'score']])
    df_output = glicko.predict(df_holdout[['student', 'question']])
    assert df_output.shape[0] == df_holdout.shape[0]
    assert 'p' in df_output.columns
    assert df_output['p'].dtype == float

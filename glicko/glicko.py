# A simply Python implemenation of the Glicko2 algorithm
# Reference document: http://www.glicko.net/glicko/glicko2.pdf
# This code uses the notation in the reference document throughout for consistency

import numpy as np
import pandas as pd

default_params = {
    'tau': .5,
    'default_rating': 1500,
    'default_RD': 350,
    'default_volatility': .06,
    'default_standardization': 173.7178,
    'tolerance': 1e-6
}

ENTITY_COL = 'entity'
ENTITY_1 = 'entity_1'
ENTITY_2 = 'entity_2'
ENTITY_RATING = 'rating'
ENTITY_RD = 'rd'
ENTITY_VOLATILITY = 'volatility'
VOLATILITY_PRIME = 'volatility_prime'
ENTITY_MU = 'mu'
ENTITY_PHI = 'phi'
NUMERIC_COLS = [ENTITY_RATING, ENTITY_RD, ENTITY_VOLATILITY, ENTITY_MU, ENTITY_PHI]
MU = 'mu'
MU_1 = 'mu_1'
MU_2 = 'mu_2'
PHI = 'phi'
PHI_1 = 'phi_1'
PHI_2 = 'phi_2'
DELTA = 'delta'
V = 'v'
SCORE_DEVIATION = 'score_deviation'
STUDENT = 'student'
QUESTION = 'question'
SCORE = 'score'

class Glicko2(object):
    def __init__(self, **kwargs):

        # Update the default parameters with kwargs, if applicable
        self.__dict__ = default_params
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        # Make dataframes for the students and questions
        # We maintain these separately for computational simplicity
        self.df_entities = pd.DataFrame(columns=
            [
                ENTITY_COL,
                ENTITY_RATING,
                ENTITY_RD,
                ENTITY_VOLATILITY,
                ENTITY_MU,
                ENTITY_PHI
            ]
        )

    def prep_df(self, df):
        df = df.rename(columns={STUDENT: ENTITY_1, QUESTION: ENTITY_2})
        df_app = df.rename(columns={ENTITY_1: ENTITY_2, ENTITY_2: ENTITY_1})
        df_app[SCORE] = df_app[SCORE].astype(int).apply(lambda x: x^1)
        df = df.append(df_app)
        return df

    def update_entities(self, df):
        current_entities = df[ENTITY_1].unique().tolist()
        old_entities = self.df_entities[ENTITY_COL].unique().tolist()
        new_entities = [c for c in current_entities if c not in old_entities]
        ratings = self.default_rating * np.ones(len(new_entities))
        rds = self.default_RD * np.ones(len(new_entities))
        volatilities = self.default_volatility * np.ones(len(new_entities))
        mus = (ratings - self.default_rating) / self.default_standardization
        phis = (rds) / self.default_standardization
        new_df = pd.DataFrame(
            {
                ENTITY_COL: new_entities,
                ENTITY_RATING: ratings,
                ENTITY_RD: rds,
                ENTITY_VOLATILITY: volatilities,
                ENTITY_MU: mus,
                ENTITY_PHI: phis
            }
        )
        self.df_entities = self.df_entities.append(new_df)

    def get_parameters(self, df):

        # Merge in the core parameters
        for entity_idx in [1, 2]:
            entity_str = ENTITY_COL + '_' + str(entity_idx)
            df = df.merge(self.df_entities, left_on=entity_str, right_on=ENTITY_COL, how='left').drop(columns=ENTITY_COL)
            df = df.rename(columns=dict(zip(NUMERIC_COLS, [N + '_' + str(entity_idx) for N in NUMERIC_COLS])))
        return df

    def get_mu_phi(self, df):
        for ii in range(0, 1):
            ii_str = str(ii)
            df[MU + '_' + ii_str] = (df[ENTITY_RATING + '_' + ii_str] - self.default_rating) / self.default_standardization
            df[PHI + '_' + ii_str] = (df[ENTITY_RD + '_' + ii_str] - self.default_rating) / self.default_standardization
        return df

    def get_v_delta(self, df):
        df['g_phi'] = 1 / np.sqrt(1 + 3*df[PHI_2]**2 / np.pi**2)
        df['p'] = 1 / (1+np.exp(-df['g_phi']*(df[MU_1] - df[MU_2])))
        df['inv_v'] = df['g_phi']**2 * df['p'] * (1-df['p'])
        df_v = df.groupby(ENTITY_COL + '_1')['inv_v'].sum().reset_index().rename(columns={'inv_v': V})
        df_v[V] = 1 / df_v[V]

        df[SCORE_DEVIATION] = df['g_phi'] * (df['score'] - df['p'])
        df_delta = df.groupby(ENTITY_1)[SCORE_DEVIATION].sum().reset_index()
        df_delta = df_delta.merge(df_v)
        df_delta[DELTA] = df_delta[SCORE_DEVIATION] * df_delta['v']
        df = df_delta[[ENTITY_1, SCORE_DEVIATION, V, DELTA]]
        df = df.rename(columns={ENTITY_1: ENTITY_COL})
        return df

    def _f(self, x, delta, phi, v, tau, a):
        e_x = np.exp(x)
        delta_2 = delta**2
        phi_2 = phi**2

        # f(x) = alpha - beta
        alpha = e_x * (delta_2 - phi_2 - v - e_x) / (2 * (phi_2 + v + e_x)**2)
        beta = (x - a) / (tau**2)

        f = alpha - beta

        return f

    def _initialize_buckets(self, sigma, delta, phi, v):

        A = np.log(sigma**2)
        a = A
        if delta**2 > phi**2 + v:
            B = np.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while self._f(A - k*self.tau, delta, phi, v, self.tau, a) < 0:
                k += 1
            B = A - k*self.tau
        return A, B


    def update_sigma_prime(self, df):
        sigma_prime = np.zeros(df.shape[0])
        for ii in range(df.shape[0]):
            row = df.iloc[ii]
            sigma, delta, phi, v = row.volatility, row.delta, row.phi, row.v
            A, B = self._initialize_buckets(sigma, delta, phi, v)
            a = np.log(sigma**2)
            f_A = self._f(A, delta, phi, v, self.tau, a)
            f_B = self._f(B, delta, phi, v, self.tau, a)
            while np.abs(B-A) > self.tolerance:
                C = A + ((A - B)*f_A / (f_B - f_A))
                f_C = self._f(C, delta, phi, v, self.tau, a)
                if f_C * f_B < 0:
                    A = B
                    f_A = f_B
                else:
                    f_A /= 2
                B = C
                f_B = f_C

            sigma_prime[ii] = np.exp(A / 2)
        df[VOLATILITY_PRIME] = sigma_prime
        return df

    def update_ratings(self, df):

        # Update phi_star with the new volatility estimate if the player is active
        df['phi_star'] = np.sqrt(df[PHI]**2 + df[VOLATILITY_PRIME]**2)

        # Update ratings and rd values
        df['phi_prime'] = 1 / np.sqrt((1/df['phi_star']**2 + 1/df['v']))
        df['mu_prime'] = df['mu'] + df['phi_prime']**2 * df['score_deviation']

        # Convert back to standard glicko rating scale
        df['r_prime'] = self.default_standardization * df['mu_prime'] + self.default_rating
        df['rd_prime'] = self.default_standardization * df['phi_prime']

        # Update the original data frame, skew the rds for the non-participating entities in the round
        dft = df[[ENTITY_COL, 'r_prime', 'rd_prime', 'phi_prime', VOLATILITY_PRIME]]
        dft = dft.rename(columns={'r_prime': ENTITY_RATING,
                                  'rd_prime': ENTITY_RD,
                                  'phi_prime': ENTITY_PHI,
                                  VOLATILITY_PRIME: ENTITY_VOLATILITY
                                  }
                        )
        self.df_entities.update(dft)

        # TODO: Finally apply the phi transformation on non-participants

        return dft

    def fit(self, dataframe):
        # dataframe has columns student, question, score

        # Step 0) Prep the df (rename student/question to entity, flip and append)
        # Also update with any new entities
        df = self.prep_df(dataframe)
        self.update_entities(df)

        # Step 1) Merge the current parameters for both entities into the new dataframe
        df = self.get_parameters(df)

        # Step 3 and 4) Compute v and delta
        df_delta_v = self.get_v_delta(df)

        # Step 5) Compute sigma' for each entity
        df_params = df_delta_v.merge(self.df_entities[
                                         [
                                             ENTITY_COL,
                                             ENTITY_MU,
                                             ENTITY_PHI,
                                             ENTITY_VOLATILITY,
                                         ]
                                     ], how='left')
        df_params = self.update_sigma_prime(df_params)

        # Step 6, 7, 8) Update the ratings and deviations
        df_params = self.update_ratings(df_params)







    def update(self, dataframe):
        pass

    def generate_synthetic_data(self, N_students, N_questions, p_obs=1):

        # Make some fake students and question data
        student_ratings = self.default_RD * np.random.randn(N_students) + self.default_rating
        student_rd = np.abs(50 * np.random.randn(N_students) + self.default_RD)
        student_volatility = .06 * np.ones(N_students)
        student_mu = (student_ratings - self.default_rating) / self.default_standardization
        student_phi = (student_rd) / self.default_standardization
        question_ratings = self.default_RD * np.random.randn(N_questions) + self.default_rating
        question_rd = np.abs(50 * np.random.randn(N_questions) + self.default_RD)
        question_volatility = .06 * np.ones(N_questions)
        question_mu = (question_ratings - self.default_rating) / self.default_standardization
        question_phi = (question_rd) / self.default_standardization

        # Convert to dfs
        df_students = pd.DataFrame(
            {
                'entity_1': ['s' + str(ii) for ii in range(0, N_students)],
                'rating_1': student_ratings,
                'rd_1': student_rd,
                'volatility_1': student_volatility,
                'mu_1': student_mu,
                'phi_1': student_phi
            }
        )

        df_questions = pd.DataFrame(
            {
                'entity_2': ['q' + str(ii) for ii in range(0, N_questions)],
                'rating_2': question_ratings,
                'rd_2': question_rd,
                'volatility_2': question_volatility,
                'mu_2': question_mu,
                'phi_2': question_phi
            }
        )

        # Cartesian merge, compute scoring outcomes
        df = df_students.assign(dummy=1).merge(df_questions.assign(dummy=1)).drop(columns='dummy')
        df['g_phi'] = df['phi_2'].apply(lambda x: 1/np.sqrt(3*x**2 / np.pi**2))
        df['p'] = 1 / (1+ np.exp(-df['g_phi'] * (df['mu_1'] - df['mu_2'])))
        df['score'] = 1.0 * (np.random.rand(df.shape[0]) < df['p'])

        # Subsample and return both the original and sampled dataframe
        df = df.rename(columns={'entity_1': 'student', 'entity_2': 'question'})
        return df.sample(frac=p_obs), df, df_students, df_questions

#Implements a IRT models up to 3PL

import numpy as np
from scipy.stats import norm
import pandas as pd
from time import time
import enum

default_params = {
    'model_name': '2pl', 'mu_theta':0, 'sigma_theta':1,
    'mu_alpha': 1, 'sigma_alpha': 1, 'mu_beta': 0,
    'sigma_beta': 1, 'alpha_gamma': 1, 'beta_gamma': 3,
    'T': 1000, 'burnin': 100, 'thinning': 1
}

class Model(enum.Enum):
    _1pl = '1pl'
    _2pl = '2pl'
    _3pl = '3pl'


class IRT3PL(object):
    def __init__(self, **kwargs):
        # Update all the parameter and add a model object based on the model_name
        self.__dict__ = default_params
        allowed_keys = default_params.keys()
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        if self.model_name==Model._3pl.value:
            self.model = Model._3pl
        if self.model_name==Model._2pl.value:
            self.model = Model._2pl
        else:
            self.model = Model._1pl
            self.sigma_alpha = 1e-6

    def convert_dataframe(self, df):

        data = df.pivot('student', 'question', 'score')
        self.data = data.values.astype(float)
        self.students = data.index.values
        self.questions = data.columns.values
        return data.values.astype(float)

    def truncnorm(self, a_t, b_t, mu_t, sigma_t):
        #Create the mask matrix for dealing with missing data (nans)
        C = np.isnan(a_t) + np.isnan(b_t) + np.isnan(mu_t) + np.isnan(sigma_t)
        idx = np.where(~C)
        a = a_t[idx]
        b = b_t[idx]
        mu = mu_t[idx]
        sigma = sigma_t[idx]
        O = np.nan * np.zeros(C.shape)
    
        N = np.prod(np.array(mu).shape)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        Phi_alpha = norm.cdf(alpha)
        Phi_beta = norm.cdf(beta)
        U = np.random.rand(N)
        out = norm.ppf(Phi_alpha + U*(Phi_beta - Phi_alpha)) * sigma + mu
    
        #If any elements in out are nan . . . if so set to mu
        nan_idx = np.where(np.isnan(out))
        out[nan_idx] = mu[nan_idx]
        O[idx] = out
        
        #Return the final result
        return O
        
    def setup_mcmc_samples(self):
        #Set up the dataframes for saving off samples
        N = self.N
        Q = self.Q
        
        samples_to_save = int((self.T - self.burnin) / self.thinning)
        self.LL = np.zeros(self.T)
        self.W_mcmc = np.zeros((N, Q))
        self.theta_mcmc = np.zeros((N, samples_to_save))
        self.beta_mcmc = np.zeros((Q, samples_to_save))
        self.alpha_mcmc = np.zeros((Q, samples_to_save))
        self.gamma_mcmc = np.zeros((Q, samples_to_save))
        
    def save_samples(self, W, Z, theta, alpha, beta, gamma, t):
        idx = int((t-self.burnin) / self.thinning)
        self.W_mcmc = self.W_mcmc + 1.0*W / ((self.T - self.burnin)/self.thinning)
        self.theta_mcmc[:, idx:idx+1] = theta
        self.beta_mcmc[:, idx:idx+1] = beta
        self.alpha_mcmc[:, idx:idx+1] = alpha
        self.gamma_mcmc[:, idx:idx+1] = gamma

    def compute_posterior_means(self):
        theta_hat = np.mean(self.theta_mcmc, axis=1)
        beta_hat = np.mean(self.beta_mcmc, axis=1)
        gamma_hat = np.mean(self.gamma_mcmc, axis=1)
        alpha_hat = np.mean(self.alpha_mcmc, axis=1)
        self.theta_df = pd.DataFrame({'student': self.students, 'theta': theta_hat})
        self.beta_df = pd.DataFrame({'question': self.questions, 'beta': beta_hat})
        self.gamma_df = pd.DataFrame({'question': self.questions, 'gamma': gamma_hat})
        self.alpha_df = pd.DataFrame({'question': self.questions, 'alpha': alpha_hat})

    def sample_W(self, Y, eta, gamma):
        if self.model == Model._3pl:
            gamma_temp = np.tile(gamma.T, (self.N, 1))
            Phi = norm.cdf(eta)
            P = Y * Phi / (gamma_temp + (1-gamma_temp)*Phi)
            W = 1.0*(np.random.rand(*P.shape) < P)
            W[np.isnan(Y)] = np.nan
            return W
        else:
            return self.data
        
    def sample_Z(self, eta, W):
        #Configure the limits based on the values in W
        A = np.zeros(W.shape)
        B = np.zeros(W.shape)
        Sigma = np.ones(W.shape)
        A[W==0] = -np.inf
        B[W==1] = np.inf
        Z = self.truncnorm(A, B, eta, Sigma)
        Z[np.isnan(W)] = np.nan
        if (np.sum(np.isinf(Z))>0):
            print("screeaaaam")
        Z[np.isinf(Z)] = np.nan
        return Z
        
    def sample_theta(self, Z, beta, alpha):
        alpha_matrix = np.tile(alpha.T, (self.N, 1))
        alpha_matrix[np.isnan(Z)] = np.nan
        alpha_square_sum = np.nansum(alpha_matrix**2, axis=1, keepdims=True)
        Z_prime = Z + np.tile(beta.T, (self.N, 1))
        Z_prime = Z_prime * np.tile(alpha.T, (self.N, 1))
        mu = (np.nansum(Z_prime, axis=1, keepdims=True) + self.mu_theta / (self.sigma_theta**2)) / (1.0/self.sigma_theta**2 + alpha_square_sum)
        var = 1.0 / (1.0/self.sigma_theta**2 + alpha_square_sum)
        theta_out = np.sqrt(var) * np.random.randn(self.N, 1) + mu
        return theta_out
    
    def sample_alpha_beta(self, theta, Z):

        xi = np.zeros((2, self.Q))
        
        Sigma_xi_inv = np.array([[1.0/self.sigma_alpha, 0],[0, 1.0/self.sigma_beta]])
        for jj in range(0, self.Q):
            valid_student_idx = np.where(~np.isnan(Z[:, jj]))[0]
            x = np.concatenate((theta[valid_student_idx, 0:1], -1*np.ones((len(valid_student_idx), 1))), axis=1)
            cov = np.linalg.inv(np.dot(x.T, x) + Sigma_xi_inv)
            post_add = np.array([self.mu_alpha, self.mu_beta])
            post_add.shape = (2, 1)
            post_add = np.dot(Sigma_xi_inv, post_add)
            mu_t = np.dot(cov, np.dot(x.T, Z[valid_student_idx, jj:jj+1]) + post_add)
            xi[:, jj] = np.random.multivariate_normal(mu_t[:,0], cov)
            
        alpha = xi[0:1,:].T
        beta = xi[1:2, :].T
        if self.model == Model._1pl:
            alpha = np.ones((self.Q, 1))
        return alpha, beta
        
    def sample_gamma(self, Y, W):
        gamma = np.zeros((self.Q, 1))
        if self.model == Model._3pl:
            a = np.nansum(1-W, axis=0)
            b = np.nansum((1-W)*Y, axis=0)
            for gg in range(0, self.Q):
                gamma[gg] = np.random.beta(self.alpha_gamma + b[gg], self.beta_gamma - b[gg] + a[gg])
        return gamma

    def predict(self, dataframe):
        # dataframe contains columns student, question
        # we will look up the posterior estimates and model to calculate a success prob and return a new df

        # Merge in the current posterior estimates . . . fillna as needed
        dataframe = dataframe[['student', 'question']]
        dataframe = dataframe.merge(self.theta_df, how='left')
        dataframe['theta'] = dataframe['theta'].fillna(self.mu_theta)
        dataframe = dataframe.merge(self.beta_df, how='left')
        dataframe['beta'] = dataframe['beta'].fillna(self.mu_beta)
        dataframe = dataframe.merge(self.alpha_df, how='left')
        dataframe['alpha'] = dataframe['alpha'].fillna(self.mu_alpha)
        dataframe = dataframe.merge(self.gamma_df, how='left')
        dataframe['gamma'] = dataframe['gamma'].fillna(0)

        # Now compute P(y=1) and return the cols that matter
        dataframe['p0'] = norm.cdf(dataframe['alpha'] * dataframe['theta'] - dataframe['beta'])
        dataframe['p'] = dataframe['gamma'] + (1-dataframe['gamma'])*dataframe['p0']
        return dataframe[['student', 'question', 'p']]

    def fit(self, dataframe, theta_init=None, beta_init=None, alpha_init=None, gamma_init=None):
        # data is a pandas dataframe of the form student_id, question_id, score
        # We'll convert this into a N x Q matrix and store off the ids for later use
        data = self.convert_dataframe(dataframe)

        self.N, self.Q = data.shape
        
        t_w = np.zeros(self.T)
        t_z = np.zeros(self.T)
        t_ab = np.zeros(self.T)
        t_th = np.zeros(self.T)
        t_g = np.zeros(self.T)
        
        #Initialize model parameters parameters according to priors
        if (theta_init is None):
            theta = self.sigma_theta * np.random.randn(self.N, 1) + self.mu_theta
        else:
            theta = theta_init
        if (beta_init is None):
            beta = self.sigma_beta * np.random.randn(self.Q, 1) + self.mu_beta
        else:
            beta = beta_init
        if (alpha_init is None):
            alpha = self.truncnorm(np.zeros((self.Q, 1)), np.inf*np.ones((self.Q, 1)), self.mu_alpha*np.ones((self.Q, 1)), self.sigma_alpha*np.ones((self.Q, 1)))
        else:
            alpha = alpha_init
        if (gamma_init is None):
            gamma = np.random.beta(self.alpha_gamma, self.beta_gamma, size=(self.Q,1))
        else:
            gamma = gamma_init
        
        #Initialize the state variables
        self.setup_mcmc_samples()
        
        #Run the chain
        for t in range(0, self.T):
            if ((t+1) % 1 == 0):
                print("Iter: " + str(t+1))

            #Compute log liklihood
            self.LL[t] = self.compute_LL(data, theta, alpha, beta, gamma)

            #Compute current value of eta = alpha*theta - beta
            eta = np.tile(alpha.T, (self.N, 1)) * np.tile(theta, (1, self.Q)) - np.tile(beta.T, (self.N, 1))
            self.eta = eta
            
            #Sample W
            t1 = time()
            W = self.sample_W(data, eta, gamma)
            t_w[t] = time() - t1
            
            #Sample Z
            t1 = time()
            Z = self.sample_Z(eta, W)
            t_z[t] = time()-t1
            
            #Sample theta
            t1 = time()
            theta = self.sample_theta(Z, beta, alpha)
            t_th[t] = time() - t1
            
            #Sample alpha, beta (jointly)
            t1 = time()
            alpha, beta = self.sample_alpha_beta(theta, Z)
            t_ab[t] = time() - t1
            
            #Sample gamma
            t1 = time()
            gamma = self.sample_gamma(data, W)
            t_g[t] = time() - t1

            #Save off values if t>burnin
            if (t >= self.burnin) & (((t-self.burnin) % self.thinning) == 0):
                self.save_samples(W, Z, theta, alpha, beta, gamma, t)
        self.t_w = t_w
        self.t_z = t_z
        self.t_g = t_g
        self.t_th = t_th
        self.t_ab = t_ab
        self.compute_posterior_means()

    def compute_LL(self, data, theta, alpha, beta, gamma):

        D = data[~np.isnan(data)]

        eta = np.tile(alpha.T, (self.N, 1)) * np.tile(theta, (1, self.Q)) - np.tile(beta.T, (self.N, 1))
        gamma_temp = np.tile(gamma.T, (self.N, 1))
        P_pos = gamma_temp + (1-gamma_temp) * norm.cdf(eta)
        P_neg = 1 - P_pos
        P_pos = P_pos[~np.isnan(data)]
        P_neg = P_neg[~np.isnan(data)]

        P = P_pos
        P[D==0] = P_neg[D==0]
        P = np.clip(P, 1e-3, 1-1e-3)

        return np.sum(-np.log(P))

        
                
    def generate_synthetic_data(self, N, Q, p_obs=1):
        
        #Generate the simple stuff
        theta = self.sigma_theta * np.random.randn(N, 1) + self.mu_theta
        beta = self.sigma_beta * np.random.randn(Q, 1) + self.mu_beta
        alpha = self.truncnorm(np.zeros((Q, 1)), np.inf*np.ones((Q, 1)), self.mu_alpha*np.ones((Q,1)), self.sigma_alpha*np.ones((Q,1)))
        gamma = np.random.beta(self.alpha_gamma, self.beta_gamma, size=(Q, 1))

        # Make modifications if 1pl or 2pl models
        # alpha = 1 for 1pl (no discrimination)
        # gamma = 0 for anything other than 3pl (no guessing)
        if self.model==Model._1pl:
            alpha = np.ones((Q, 1))
        if self.model!=Model._3pl:
            gamma = np.zeros((Q, 1))
        
        #Sample p(Y|-)
        G = np.tile(gamma.T, (N, 1))
        eta = np.tile(alpha.T, (N, 1)) * np.tile(theta, (1, Q)) - np.tile(beta.T, (N, 1))
        P = norm.cdf(eta) + G*(1-norm.cdf(eta))
        Y = 1.0*(np.random.rand(N, Q) < P)
        
        # Reshape everything into dataframes, sample, and return
        student_ids = ['S' + str(ii) for ii in range(0, N)]
        question_ids = ['Q' + str(ii) for ii in range(0, Q)]
        Y_df = pd.DataFrame(Y, columns=question_ids)
        Y_df['student'] = student_ids
        Y_df = pd.melt(Y_df, 'student', var_name='question', value_name='score')
        theta_df = pd.DataFrame({'student': student_ids, 'theta': theta[:, 0]})
        beta_df = pd.DataFrame({'question': question_ids, 'beta': beta[:, 0]})
        alpha_df = pd.DataFrame({'question': question_ids, 'alpha': alpha[:, 0]})
        gamma_df = pd.DataFrame({'questin': question_ids, 'gamma': gamma[:, 0]})

        return Y_df.sample(frac=p_obs), Y_df, theta_df, beta_df, alpha_df, gamma_df


# Want to examine stability under the local update conditions
# To do this, generate some synthetic data, and do a single batch fit to get initial estimates
# Then, try premuting the data order and doing a series of local updates
# After each series of mini-batches, compute estimate deviaance
# Compile and display results

from glicko import Glicko2
from scipy.stats import kendalltau
from plotnine import *
import pandas as pd
import numpy as np

# Some params
N_students = 50
N_questions = 50
p_obs = .1
num_runs = 5

# Generate synthetic data and do a full batch fit
glicko_batch = Glicko2()
df, df_full, _, _ = glicko_batch.generate_synthetic_data(N_students, N_questions, p_obs)
df_questions = df.copy()
df_questions = df_questions[['question', 'rating_2', 'rd_2', 'volatility_2', 'mu_2', 'phi_2']].drop_duplicates()
df_questions = df_questions.rename(columns={'question': 'entity'})
cols = df_questions.columns.values.tolist()
cols = [c.replace('_2', '') for c in cols]
df_questions.columns = cols

df_item = df.copy()[['student', 'rating_1']].drop_duplicates().rename(columns={'student': 'entity', 'rating_1': 'rating_true'})
df_student = df_item.copy()

# Now create a new Glicko object and do mini-batch updates (one row at a time)
# Do this for num_runs permutations of the dataframe
for nn in range(num_runs):
    df_perm = df.sample(frac=1)
    glicko_mini = Glicko2(update_questions=False)
    glicko_mini.df_entities = df_questions
    for row_idx in range(df_perm.shape[0]):
        print("{} : {}/{}]".format(nn, row_idx, df_perm.shape[0]))
        glicko_mini.fit(df_perm[['student', 'question', 'score']].iloc[row_idx:row_idx+1])
    df_out = glicko_mini.df_entities[['entity', 'rating']].rename(columns={'rating': 'rating_'+str(nn)})
    df_item = df_item.merge(df_out)

# Now permute on student -- each student will be processed as a batch of interactions
students = df_student['entity'].unique()
for nn in range(num_runs):
    print(nn)
    glicko_mini = Glicko2(update_questions=False)
    glicko_mini.df_entities = df_questions
    for student in np.random.choice(students, len(students), replace=False):
        df_set = df[df['student']==student]
        glicko_mini.fit(df_set[['student', 'question', 'score']])
    df_out = glicko_mini.df_entities[['entity', 'rating']].rename(columns={'rating': 'rating_'+str(nn)})
    df_student = df_student.merge(df_out)

# Get standard deviation estimates
sigma = df['rating_1'].std()
d_item = np.std(df_item.values[:, 1:].astype(float), axis=1) / sigma
d_student = np.std(df_student.values[:, 1:].astype(float), axis=1) / sigma
df_sigma = pd.DataFrame({'idx': range(0, len(d_item)), 'type': ['item']*len(d_item), 'sigma': d_item})
df_sigma = df_sigma.append(pd.DataFrame({'idx': range(0, len(d_student)), 'type': ['student']*len(d_student), 'sigma': d_student}))

# Get kendall tau values between all the ratings and true scores
tau_item = [kendalltau(df_item['rating_true'], df_item['rating_'+str(ii)])[0] for ii in range(0, num_runs)]
tau_student = [kendalltau(df_student['rating_true'], df_student['rating_'+str(ii)])[0] for ii in range(0, num_runs)]
df_tau = pd.DataFrame({'idx': range(0, len(tau_item)), 'type': ['item']*len(tau_item), 'tau': tau_item})
df_tau = df_tau.append(pd.DataFrame({'idx': range(0, len(tau_student)), 'type': ['student']*len(tau_student), 'tau': tau_student}))

# Do the plots
fig_sigma_density = ggplot(df_sigma, aes('sigma', colour='type')) + geom_density() + xlab('d')
fig_tau_density = ggplot(df_tau, aes('tau', colour='type')) + geom_density() + xlab("Kendall's tau")
ggsave(plot=fig_sigma_density, filename='sigma_density_pretrain.png', dpi=1000)
ggsave(plot=fig_tau_density, filename='tau_density_pretrain.png', dpi=1000)









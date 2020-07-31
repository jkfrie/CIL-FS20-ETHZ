""" Baseline Model 1: Simply predict everything as background """

import pandas as pd

MODEL_NAME = 'baseline1_model'
df = pd.read_csv('sample_submission.csv', header=0)
df['Prediction'].values[:] = 0
df.to_csv('./Results/Submissions/{}.csv'.format(MODEL_NAME), index=False)
import psycopg2 as pg
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sqlalchemy import create_engine

ALERT_AGENT_THRESHOLD = .15
LOCK_ACCOUNT_THRESHOLD = .3

connection = pg.connect(
    host='localhost',
    port=54320,
    dbname='ht_db',
    user='postgres'
)
engine = create_engine('postgresql://postgres:password@127.0.0.1:54320/ht_db')

data_path = '../data'
model_path = 'artifacts'
schema_path = '../misc/schemas.yaml'

# LGBM RandomSearch parameters
param_test = {'num_leaves': sp_randint(6, 50),
              'subsample': sp_uniform(loc=0.2, scale=0.8),
              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
              'min_data_in_leaf': sp_randint(100, 3000),
              'max_bin': sp_randint(150, 400),
              'scale_pos_weight': sp_randint(2, 90)}

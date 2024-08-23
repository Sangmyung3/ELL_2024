## Inflation Forecast
## Target variable: Year over Year inflation
import os
import sqlite3
import numpy as np
import pandas as pd
import sqlite3
from functions import get_data,get_Gram_rbf
import xgboost as xgb

Transformation = 'Transform' 
Target = 'Inflation'

price_var = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 
             'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
             'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']
lags = 12

X_used, Y_used, Date_used = get_data(Transformation, lags)

n = X_used.shape[0]

forecast_period = pd.to_datetime('2015-01-01')<=Date_used
forecast_idx = np.where(forecast_period)[0]
n_test = np.sum(forecast_period)

validation_period = (pd.to_datetime('2005-08-01')<=Date_used) & (pd.to_datetime('2015-01-01') > Date_used)
validation_idx = np.where(validation_period)[0]
n_val = np.sum(validation_period)

training_period = pd.to_datetime('2005-08-01')>Date_used
training_idx = np.where(training_period)[0]
n_train = np.sum(training_period)

X_train = X_used.loc[training_period,:]
Y_train = Y_used.loc[training_period]

X_val = X_used.loc[validation_period,:]
Y_val = Y_used.loc[validation_period]

X_test = X_used.loc[forecast_period,:]
Y_test = Y_used.loc[forecast_period]

CV_grid_n = 5

############################################################################
############################  basic fetch  ##################################
############################################################################
database_name = 'database_predict_inflation.db'

con = sqlite3.connect(os.path.join('Results', database_name))
cur = con.cursor()

res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
table_names = res.fetchall()

res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
res.fetchone()

cur.close()
con.close()

############################################################################
############################################################################
database_name = 'database_example.db'

con = sqlite3.connect(os.path.join('Results', database_name))
cur = con.cursor()

res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
table_names = res.fetchall()

if ~np.isin('Results', table_names):
    print("CREATE NEW DATABASE")
    cur.execute("""CREATE TABLE IF NOT EXISTS Results(
                Model TEXT NOT NULL,
                Target TEXT NOT NULL,
                Transformation TEXT NOT NULL,
                Seed INTEGER NOT NULL,
                Parameter_1 TEXT,
                Parameter_2 TEXT,
                Window_size INTEGER NOT NULL,
                Validation_size INTEGER NOT NULL,
                RMSE REAL NOT NULL,
                PRIMARY KEY (Model, Target, Transformation, Seed, Window_size, Validation_size))""")
    con.commit()
else:
    print("DATABASE ALREADY EXISTS")

num_parallel_tree = 5
subsample = np.sqrt(X_train.shape[0])/X_train.shape[0]  # 23 samples
n_estimators_list = np.linspace(1,50,CV_grid_n).astype(int)
# Seed number used
seed_list = [42, 43, 44, 45, 46]
RMSE_XGBs = []
param_XGBs = []
for seed in seed_list:

    val_err = np.zeros((n_val,len(n_estimators_list)))
    XGBmodel_dict = {}
    for cv_i, n_estimators in enumerate(n_estimators_list):
        XGBmodel_dict[cv_i] = xgb.XGBRegressor(n_jobs=1, tree_method="exact", subsample=subsample,
                                                num_parallel_tree=num_parallel_tree,
                                                n_estimators=n_estimators, random_state=seed)
        XGBmodel_dict[cv_i].fit(X_train, Y_train)
        Y_hat = XGBmodel_dict[cv_i].predict(X_val)
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    Y_hat = XGBmodel_dict[min_idx].predict(X_test)
    param_XGBs.append(n_estimators_list[min_idx])
    RMSE_XGBs.append(np.sqrt(np.mean((Y_test.values - Y_hat)**2)))

table = pd.DataFrame((param_XGBs,RMSE_XGBs)).T
table.columns = ['Parameter_1', 'XGBoost-subsample']
table['Seed'] = seed_list
table_XGB_melt = pd.melt(table,id_vars=['Seed','Parameter_1'],value_vars=['XGBoost-subsample'])
table_XGB_melt = table_XGB_melt.rename(columns = {'variable':'Model','value':'RMSE'})
table_XGB_melt['Target'] = Target
table_XGB_melt['Transformation'] = Transformation
table_XGB_melt['Parameter_2'] = None
table_XGB_melt['Window_size'] = n_train
table_XGB_melt['Validation_size'] = n_val
table_XGB_melt = table_XGB_melt[['Model','Target','Transformation','Seed','Parameter_1','Parameter_2','Window_size','Validation_size','RMSE']]

query = ''' insert or replace into Results (Model,Target,Transformation,Seed,Parameter_1,Parameter_2,Window_size,Validation_size,RMSE) values (?,?,?,?,?,?,?,?,?) '''
cur.executemany(query, table_XGB_melt.values)
con.commit()
con.close()

################################################################################################################################
################################################################################################################################
################################################################################################################################

database_name = 'database_example.db'

con = sqlite3.connect(os.path.join('Results', database_name))
query = """SELECT * FROM RESULTS where Model='%s' and Transformation='%s' and Target='%s'""" %('XGBoost-subsample',Transformation,Target)
RESULTS = pd.read_sql(query, con)

print(RESULTS.pivot(index=['Seed'], columns='Model', values=['RMSE','Parameter_1']))

## Inflation Forecast
## Target variable: Year over Year inflation
import os
import time
import pickle
import sqlite3
import numba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import sqlite3
import random
from functions import get_data,get_Gram_rbf
import xgboost as xgb
import tensorflow as tf

# Transform:
# Transformation = 'No Transform'
Transformation = 'Transform' 
# :Transforms according to the recommendations given by McCracken and Ng (2015) for all but Group 7 (Prices),
#  which are transformed as year over year growth

# Target = 'Inflation'
Target = 'Inflation MoM'

price_var = ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 
             'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
             'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']

if Target =='Inflation':
    lags = 12
elif Target =='Inflation MoM':
    lags = 1

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

nnan_idx = np.sum(np.isnan(X_used), axis=0)==0
X_used_nnan = X_used.loc[:,nnan_idx]

X_train_nnan = X_used_nnan.loc[training_period,:]
X_val_nnan = X_used_nnan.loc[validation_period,:]
X_test_nnan = X_used_nnan.loc[forecast_period,:]

plt.plot(pd.to_datetime(Date_used),Y_used)
plt.title("Inflation: $\log CPI_t-\log CPI_{t-12}$")
# plt.show()
plt.savefig(os.path.join('Figures', 'inflation_yoy.png'))
plt.close()

CV_grid_n = 30

# Seed number used
seed_list = [42, 43, 44, 45, 46]
database_name = 'database_predict_inflation.db'
for seed in seed_list:
    
    # Make Database
    con = sqlite3.connect(os.path.join('Results', database_name))
    cur = con.cursor()

    res = cur.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
    table_names = res.fetchall()
    if ~np.isin('Results', table_names):
        print("CREATE NEW DATABASE")
        cur.execute("""CREATE TABLE IF NOT EXISTS Results(
                    Date TEXT NOT NULL,
                    Target TEXT NOT NULL,
                    Value REAL NOT NULL,
                    Prediction REAL NOT NULL,
                    Model TEXT NOT NULL,
                    Seed INTEGER NOT NULL,
                    Parameter TEXT,
                    Window_size INTEGER NOT NULL,
                    Validation_size INTEGER NOT NULL,
                    Transformation TEXT NOT NULL,
                    PRIMARY KEY (Date, Target, Model, Seed, Window_size, Validation_size, Transformation))""")
        con.commit()
    else:
        print("DATABASE ALREADY EXISTS")


    if ~np.isin('CV_Error', table_names):
        print("CREATE NEW DATABASE TABLE")
        str_append = ''
        for s in np.arange(1,CV_grid_n+1):
            str_append = str_append+'col_%i REAL, '%s

        cur.execute("""CREATE TABLE IF NOT EXISTS CV_Error(
                    Model TEXT NOT NULL,
                    Target TEXT NOT NULL,
                    Tune_Param TEXT NOT NULL,
                    Seed INTEGER NOT NULL,
                    Transformation TEXT NOT NULL,
                    Window_size INTEGER NOT NULL,
                    Validation_size INTEGER NOT NULL,"""\
                    + str_append\
                    + """PRIMARY KEY (Model, Target, Tune_Param, Seed, Transformation, Window_size, Validation_size))""")
        con.commit()
    else:
        print("DATABASE TABLE ALREADY EXISTS")
    #######################################################################################
    ################################   AR(1), AR(12)  #####################################
    #######################################################################################
    OLS = LinearRegression(fit_intercept=True)
    OLS.fit(X_train[['CPIAUCSL']], Y_train)
    # OLS.coef_

    Y_hat = OLS.predict(X_test[['CPIAUCSL']])
    RMSE_AR1 = np.sqrt(np.mean((Y_test-Y_hat)**2))

    AR1_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'AR1',
            'Seed': seed,
            'Parameter': '',
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    AR1_out = pd.DataFrame.from_dict(AR1_out)

    lags_label = np.append(['CPIAUCSL'],['CPIAUCSL_lag%i' % p for p in range(1,12)])

    OLS = LinearRegression(fit_intercept=True)
    OLS.fit(X_train[lags_label], Y_train)
    # OLS.coef_

    Y_hat = OLS.predict(X_test[lags_label])
    RMSE_AR12 = np.sqrt(np.mean((Y_test-Y_hat)**2))

    AR12_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'AR12',
            'Seed': seed,
            'Parameter': '',
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    AR12_out = pd.DataFrame.from_dict(AR12_out)

    OLS = LinearRegression(fit_intercept=True)
    OLS.fit(X_train[['CPIAUCSL','CPIAUCSL_lag11']], Y_train)
    # OLS.coef_

    Y_hat = OLS.predict(X_test[['CPIAUCSL','CPIAUCSL_lag11']])
    RMSE_AR1_12 = np.sqrt(np.mean((Y_test-Y_hat)**2))

    AR1_12_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'AR1_12',
            'Seed': seed,
            'Parameter': '',
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    AR1_12_out = pd.DataFrame.from_dict(AR1_12_out)


    #######################################################################################
    ################################   Rolling Average  ###################################
    #######################################################################################
    average_n_list = np.linspace(1,50,CV_grid_n).astype(int)

    val_err = np.zeros((n_val, len(average_n_list)))
    Y_long = pd.concat((Y_train, Y_val),axis=0)
    for i in range(len(Y_val)):
        for cv_i, average_n in enumerate(average_n_list):
            Y_hat = np.mean(Y_long.iloc[i+n_train-average_n:i+n_train])
            val_err[i, cv_i] = Y_val.iloc[i] - Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_RA = np.mean(np.array(val_err)**2, axis=0)

    RA_VE_out = {'Model':['RA']*2,
                 'Target':[Target]*2,
                'Tune_Param': ['CV_grid','RA_lags'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        try:
            RA_VE_out[cn] = [average_n_list[i], val_err_RA[i]]
        except:
            RA_VE_out[cn] = [None, None]

    RA_VE_out = pd.DataFrame(RA_VE_out)

    # plt.plot(average_n_list, val_err_RA)
    # plt.xlabel('average number')
    # plt.title('Validation Error, Random Forest, Minimum=%s'%str(average_n_list[min_idx]))
    # # plt.savefig("Figures/RF_validation_seed%i.png"%seed)
    # # plt.close()
    # plt.show()

    # test_err = np.zeros((n_test, len(average_n_list)))
    # Y_long = pd.concat((Y_val, Y_test),axis=0)
    # Y_hat = np.zeros((n_test,))
    # for i in range(len(Y_test)):
    #     for cv_i, average_n in enumerate(average_n_list):
    #         Y_hat[i] = np.mean(Y_long.iloc[i+n_val-average_n:i+n_val])
    #         test_err[i, cv_i] = Y_test.iloc[i] - Y_hat[i]
    # RMSE_RA = np.sqrt(np.mean(np.array(test_err)**2, axis=0))[min_idx]

    average_n = average_n_list[min_idx]
    test_err = np.zeros((n_test,))
    Y_long = pd.concat((Y_val, Y_test), axis=0)
    Y_hat = np.zeros((n_test,))
    for i in range(len(Y_test)):
        Y_hat[i] = np.mean(Y_long.iloc[i+n_val-average_n:i+n_val])
        test_err[i] = Y_test.iloc[i] - Y_hat[i]
    
    RMSE_RA = np.sqrt(np.mean(np.array(test_err)**2, axis=0))

    RA_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'RA',
            'Seed': seed,
            'Parameter': str(average_n_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RA_out = pd.DataFrame.from_dict(RA_out)
    


    #######################################################################################
    ################################   AR, Validate  #####################################
    #######################################################################################
    ar_p_list = np.arange(1,12)

    val_err = np.zeros((n_val, len(ar_p_list)))
    AR_dict = {}
    for cv_i, ar_p in enumerate(ar_p_list):

        lags_label = np.append(['CPIAUCSL'],['CPIAUCSL_lag%i' % p for p in range(1,ar_p)])

        AR_dict[cv_i] = LinearRegression(fit_intercept=True)
        AR_dict[cv_i].fit(X_train[lags_label], Y_train)
        # OLS.coef_

        Y_hat = AR_dict[cv_i].predict(X_val[lags_label])
        val_err[:, ar_p-1] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_AR = np.mean(np.array(val_err)**2, axis=0)

    AR_VE_out = {'Model':['AR']*2,
                 'Target':[Target]*2,
                'Tune_Param': ['CV_grid','AR_lags'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        try:
            AR_VE_out[cn] = [ar_p_list[i], val_err_AR[i]]
        except:
            AR_VE_out[cn] = [None, None]

    AR_VE_out = pd.DataFrame(AR_VE_out)

    plt.plot(ar_p_list,val_err_AR)
    plt.xlabel('number of lags')
    plt.title('Validation Error, AR, argmin=%i'%ar_p_list[min_idx])
    plt.savefig("Figures/AR_validation_seed%i.png"%seed)
    plt.close()
    # plt.show()

    lags_label = np.append(['CPIAUCSL'],['CPIAUCSL_lag%i' % p for p in range(1,ar_p_list[min_idx])])

    Y_hat = AR_dict[min_idx].predict(X_test[lags_label])
    test_err_AR = Y_test.values - Y_hat
    RMSE_AR = np.sqrt(np.sum(test_err_AR**2)/len(test_err_AR))

    AR_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'AR',
            'Seed': seed,
            'Parameter': str(ar_p_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    AR_out = pd.DataFrame.from_dict(AR_out)

    #######################################################################################
    ################################   Random Walk  #######################################
    #######################################################################################
    Y_hat = X_test['CPIAUCSL']
    RMSE_RW = np.sqrt(np.mean((Y_test-Y_hat)**2))

    RW_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat.values.reshape((-1,)),
            'Model': 'Random Walk',
            'Seed': seed,
            'Parameter': '',
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RW_out = pd.DataFrame.from_dict(RW_out)

    #######################################################################################
    ################################   Random Forest  #####################################
    #######################################################################################
    
    max_depth_list = np.append(np.linspace(1,50,CV_grid_n-1).astype(int),None)
    
    val_err = np.zeros((n_val, len(max_depth_list)))
    RFmodel_dict = {}
    for cv_i, max_depth in enumerate(max_depth_list):
        RFmodel_dict[cv_i] = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                        max_depth=max_depth, min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                        random_state=seed, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)    
        RFmodel_dict[cv_i].fit(X_train_nnan, Y_train)
        Y_hat = RFmodel_dict[cv_i].predict(X_val_nnan)
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_RF = np.mean(np.array(val_err)**2, axis=0)

    temp_grid = ['None' if x==None else x for x in max_depth_list]
    RF_VE_out = {'Model':['Random Forest']*2,
                 'Target':[Target]*2,
                'Tune_Param': ['CV_grid','max_depth'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        RF_VE_out[cn] = [temp_grid[i], val_err_RF[i]]

    RF_VE_out = pd.DataFrame(RF_VE_out)

    plt.plot(temp_grid, val_err_RF)
    plt.xlabel('max_depth')
    plt.title('Validation Error, Random Forest, argmin=%s'%str(temp_grid[min_idx]))
    plt.savefig("Figures/RF_validation_seed%i.png"%seed)
    plt.close()
    # plt.show()

    Y_hat = RFmodel_dict[min_idx].predict(X_test_nnan)
    test_err_RF = Y_test.values - Y_hat
    RMSE_RF = np.sqrt(np.sum(test_err_RF**2)/len(test_err_RF))

    RF_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'Random Forest',
            'Seed': seed,
            'Parameter': str(max_depth_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RF_out = pd.DataFrame.from_dict(RF_out)

        
    #######################################################################################
    ###################################   XGBoost  ########################################
    #######################################################################################
    n_estimators_list = np.linspace(1,50,CV_grid_n).astype(int)
    val_err = np.zeros((n_val,len(n_estimators_list)))
    XGBmodel_dict = {}
    for cv_i, n_estimators in enumerate(n_estimators_list):
        XGBmodel_dict[cv_i] = xgb.XGBRegressor(n_jobs=1, tree_method="exact", n_estimators=n_estimators, random_state=seed)
        XGBmodel_dict[cv_i].fit(X_train_nnan, Y_train)
        Y_hat = XGBmodel_dict[cv_i].predict(X_val_nnan) 
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_XGB = np.mean(np.array(val_err)**2, axis=0)

    XGB_VE_out = {'Model':['XGBoost']*2,
                  'Target':[Target]*2,
                'Tune_Param': ['CV_grid','n_estimators'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        XGB_VE_out[cn] = [n_estimators_list[i], val_err_XGB[i]]

    XGB_VE_out = pd.DataFrame(XGB_VE_out)

    plt.plot(n_estimators_list,val_err_XGB)
    plt.xlabel('n_estimators')
    plt.title('Validation Error, XGBoost, argmin=%i'%n_estimators_list[min_idx])
    plt.savefig("Figures/XGB_validation_seed%i.png"%seed)
    plt.close()
    # # plt.show()

    Y_hat = XGBmodel_dict[min_idx].predict(X_test_nnan)
    test_err_XGB = Y_test.values - Y_hat
    RMSE_XGB = np.sqrt(np.sum(test_err_XGB**2)/len(test_err_XGB))

    XGB_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'XGBoost',
            'Seed': seed,
            'Parameter': str(n_estimators_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    XGB_out = pd.DataFrame.from_dict(XGB_out)

    #######################################################################################
    ##############################   XGBoost with subsampling #############################
    #######################################################################################
    
    num_parallel_tree = 100
    subsample = np.sqrt(X_train_nnan.shape[0])/X_train_nnan.shape[0]
    n_estimators_list = np.linspace(1,50,CV_grid_n).astype(int)
    val_err = np.zeros((n_val,len(n_estimators_list)))
    XGBmodel_dict = {}
    for cv_i, n_estimators in enumerate(n_estimators_list):
        XGBmodel_dict[cv_i] = xgb.XGBRegressor(n_jobs=1, tree_method="exact", subsample=subsample,
                                                num_parallel_tree=num_parallel_tree,
                                                n_estimators=n_estimators, random_state=seed)
        XGBmodel_dict[cv_i].fit(X_train_nnan, Y_train)
        Y_hat = XGBmodel_dict[cv_i].predict(X_val_nnan)
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_XGBs = np.mean(np.array(val_err)**2, axis=0)


    XGBs_VE_out = {'Model':['XGBoost-subsample']*2,
                   'Target':[Target]*2,
                    'Tune_Param': ['CV_grid','n_estimators'],
                    'Seed':[seed]*2,
                    'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        XGBs_VE_out[cn] = [n_estimators_list[i], val_err_XGBs[i]]

    XGBs_VE_out = pd.DataFrame(XGBs_VE_out)

    plt.plot(n_estimators_list, val_err_XGBs)
    plt.xlabel('n_estimators')
    plt.title('Validation Error, XGBoost with subsample, argmin=%i'%n_estimators_list[min_idx])
    plt.savefig("Figures/XGBs_validation_seed%i.png"%seed)
    plt.close()
    # # plt.show()

    Y_hat = XGBmodel_dict[min_idx].predict(X_test_nnan)
    test_err_XGBs = Y_test.values - Y_hat
    RMSE_XGBs = np.sqrt(np.sum(test_err_XGBs**2)/len(test_err_XGBs))

    XGBs_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'XGBoost-subsample',
            'Seed': seed,
            'Parameter': str(n_estimators_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    XGBs_out = pd.DataFrame.from_dict(XGBs_out)

    #######################################################################################
    ###################################   Ridge    ########################################
    #######################################################################################
    X_train_stzd = (X_train_nnan - np.mean(X_train_nnan, axis=0))/np.std(X_train_nnan, axis = 0)
    X_val_stzd = (X_val_nnan - np.mean(X_train_nnan, axis=0))/np.std(X_train_nnan, axis = 0)
    X_test_stzd = (X_test_nnan - np.mean(X_train_nnan, axis=0))/np.std(X_train_nnan, axis = 0)    

    alpha_list = np.linspace(0,10000,CV_grid_n)
    val_err = np.zeros((n_val,len(alpha_list)))
    Ridgemodel_dict = {}
    for cv_i, alpha in enumerate(alpha_list):
        Ridgemodel_dict[cv_i] = Ridge(alpha=alpha,fit_intercept=True, random_state=seed)
        Ridgemodel_dict[cv_i].fit(X_train_stzd, Y_train)
        Y_hat = Ridgemodel_dict[cv_i].predict(X_val_stzd) 
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    min_idx_ridge = min_idx
    val_err_Ridge = np.mean(np.array(val_err)**2, axis=0)

    Ridge_VE_out = {'Model':['Ridge']*2,
                    'Target':[Target]*2,
                'Tune_Param': ['CV_grid','alpha'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        Ridge_VE_out[cn] = [alpha_list[i], val_err_Ridge[i]]

    Ridge_VE_out = pd.DataFrame(Ridge_VE_out)


    plt.plot(alpha_list,val_err_Ridge)
    plt.xlabel('alpha')
    plt.title('Validation Error, Ridge, argmin=%0.2f'%alpha_list[min_idx])
    # plt.show()
    plt.savefig("Figures/Ridge_validation_seed%i.png"%seed)
    plt.close()

    
    Y_hat = Ridgemodel_dict[min_idx].predict(X_test_stzd)
    test_err_Ridge = Y_test.values - Y_hat
    RMSE_Ridge = np.sqrt(np.sum(test_err_Ridge**2)/len(test_err_Ridge))

    Ridge_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'Ridge',
            'Seed': seed,
            'Parameter': str(alpha_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    Ridge_out = pd.DataFrame.from_dict(Ridge_out)

    #######################################################################################
    ###################################   LASSO    ########################################
    #######################################################################################
    alpha_list = np.linspace(1e-15,0.001,CV_grid_n)
    val_err = np.zeros((n_val,len(alpha_list)))
    Lassomodel_dict = {}
    for cv_i, alpha in enumerate(alpha_list):
        Lassomodel_dict[cv_i] = Lasso(alpha=alpha, fit_intercept=True,  warm_start=True, random_state=seed)
        Lassomodel_dict[cv_i].fit(X_train_stzd, Y_train)
        Y_hat = Lassomodel_dict[cv_i].predict(X_val_stzd) 
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_Lasso = np.mean(np.array(val_err)**2, axis=0)

    Lasso_VE_out = {'Model':['LASSO']*2,
                    'Target':[Target]*2,
                'Tune_Param': ['CV_grid','alpha'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        Lasso_VE_out[cn] = [alpha_list[i], val_err_Lasso[i]]

    Lasso_VE_out = pd.DataFrame(Lasso_VE_out)

    plt.plot(alpha_list, val_err_Lasso)
    plt.xlabel('alpha')
    plt.title('Validation Error, LASSO, argmin=%0.7f'%alpha_list[min_idx])
    plt.savefig("Figures/LASSO_validation_seed%i.png"%seed)
    plt.close()
    # plt.show()
    # X_train_stzd.columns[Lassomodel_dict[min_idx].coef_ !=0]
    # Lassomodel_dict[min_idx].coef_[Lassomodel_dict[min_idx].coef_ !=0]

    Y_hat = Lassomodel_dict[min_idx].predict(X_test_stzd)
    test_err_Lasso = Y_test.values - Y_hat
    RMSE_Lasso = np.sqrt(np.sum(test_err_Lasso**2)/len(test_err_Lasso))

    Lasso_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'LASSO',
            'Seed': seed,
            'Parameter': str(alpha_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    Lasso_out = pd.DataFrame.from_dict(Lasso_out)

    #######################################################################################
    ##############################   ADALASSO (BIC,AIC)   #################################
    #######################################################################################
    X_train_stzd_tilde = X_train_stzd*np.abs(Ridgemodel_dict[min_idx_ridge].coef_)
    X_val_stzd_tilde = X_val_stzd*np.abs(Ridgemodel_dict[min_idx_ridge].coef_)
    X_test_stzd_tilde = X_test_stzd*np.abs(Ridgemodel_dict[min_idx_ridge].coef_)

    alpha_list = np.linspace(1e-15,0.0000001,50)
    val_err = np.zeros((n_val,len(alpha_list)))
    AIC = np.zeros((len(alpha_list),))
    BIC = np.zeros((len(alpha_list),))
    AdaLassomodel_dict = {}
    for cv_i, alpha in enumerate(alpha_list):
        AdaLassomodel_dict[cv_i] = Lasso(alpha=alpha, fit_intercept=True,  warm_start=True, random_state=seed)
        AdaLassomodel_dict[cv_i].fit(X_train_stzd_tilde, Y_train)

        DF = np.sum(AdaLassomodel_dict[cv_i].coef_ !=0)
        RSS = np.sum((Y_train-AdaLassomodel_dict[cv_i].predict(X_train_stzd_tilde))**2)
        BIC[cv_i] = n_train*np.log(RSS) + DF*np.log(n_train)
        AIC[cv_i] = n_train*np.log(RSS) + DF*2

        Y_hat = AdaLassomodel_dict[cv_i].predict(X_val_stzd_tilde) 
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_AdaLasso = np.mean(np.array(val_err)**2, axis=0)

    min_idx_AIC = np.argmin(AIC)
    min_idx_BIC = np.argmin(BIC)
    
    AdaLasso_VE_out = {'Model':['ADALASSO']*2,
                       'Target':[Target]*2,
                'Tune_Param': ['CV_grid','alpha'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        AdaLasso_VE_out[cn] = [alpha_list[i], val_err_AdaLasso[i]]

    AdaLasso_VE_out = pd.DataFrame(AdaLasso_VE_out)

    plt.plot(alpha_list, val_err_AdaLasso)
    plt.xlabel('alpha')
    plt.title('Validation Error, ADALASSO, argmin=%0.7f'%alpha_list[min_idx])
    plt.savefig("Figures/ADALASSO_validation_seed%i.png"%seed)
    plt.close()
    # plt.show()

    Y_hat = AdaLassomodel_dict[min_idx].predict(X_test_stzd_tilde)
    test_err_Lasso = Y_test.values - Y_hat
    RMSE_AdaLasso = np.sqrt(np.sum(test_err_Lasso**2)/len(test_err_Lasso))

    AdaLasso_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'ADALASSO',
            'Seed': seed,
            'Parameter': str(alpha_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    AdaLasso_out = pd.DataFrame.from_dict(AdaLasso_out)
    
    #######################################################################################
    ####################################   PCR    #########################################
    #######################################################################################
    Sigma_hat = X_train_stzd.T@X_train_stzd/n_train
    eigval, eigvec = np.linalg.eigh(Sigma_hat)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    idx = eigval.argsort()[::-1]
    eigval_sorted = eigval[idx]
    eigvec_sorted = eigvec[:, idx]
    F_train = X_train_stzd @ eigvec_sorted
    F_val = X_val_stzd @ eigvec_sorted
    F_val.columns = F_train.columns
    F_test = X_test_stzd @ eigvec_sorted
    F_test.columns = F_test.columns
    # fig, ax = plt.subplots()
    # ax.plot(X_train_stzd.columns,eigvec_sorted[:,0], label='First basis')
    # ax.plot(X_train_stzd.columns,eigvec_sorted[:,1], label='Second basis')
    # ax.plot(X_train_stzd.columns,eigvec_sorted[:,2], label='Third basis')
    # plt.xticks(rotation=-45)
    # plt.legend()
    # plt.show()


    nfactors_list = np.linspace(1,200,CV_grid_n).astype(int)
    
    val_err = np.zeros((n_val, len(nfactors_list)))
    OLS_dict = {}
    for cv_i, nfactors in enumerate(nfactors_list):
        OLS = LinearRegression(fit_intercept=True)
        OLS_dict[cv_i] = OLS.fit(F_train.iloc[:,:nfactors], Y_train)
        Y_hat = OLS_dict[cv_i].predict(F_val.iloc[:,:nfactors])
        val_err[:, cv_i] = Y_val.values-Y_hat

    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_PCR = np.mean(np.array(val_err)**2, axis=0)

    PCR_VE_out = {'Model':['PCR']*2,
                  'Target':[Target]*2,
                'Tune_Param': ['CV_grid','nfactors'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        PCR_VE_out[cn] = [nfactors_list[i], val_err_PCR[i]]

    PCR_VE_out = pd.DataFrame(PCR_VE_out)

    plt.plot(nfactors_list, val_err_PCR)
    plt.xlabel('Number of principal components')
    plt.title('Validation Error, PCR,Minimum=%i'%nfactors_list[min_idx])
    # plt.show()
    plt.savefig("Figures/PCR_validation_seed%i.png"%seed)
    plt.close()
    # # plt.show()

    Y_hat = OLS_dict[min_idx].predict(F_test.iloc[:,:nfactors_list[min_idx]])
    test_err_PCR = Y_test.values - Y_hat
    RMSE_PCR = np.sqrt(np.sum(test_err_PCR**2)/len(test_err_PCR))

    PCR_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'PCR',
            'Seed': seed,
            'Parameter': str(nfactors_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    PCR_out = pd.DataFrame.from_dict(PCR_out)

    #######################################################################################
    ###################################  RKHS regression  #################################
    #######################################################################################

    gamma = 1/X_train_stzd.shape[1]
    Kernel_Gram = get_Gram_rbf(X_train_stzd.values,X_train_stzd.values, n_train, n_train, gamma)
    K_val = get_Gram_rbf(X_train_stzd.values, X_val_stzd.values, n_train, n_val, gamma)
    K_test = get_Gram_rbf(X_train_stzd.values, X_test_stzd.values, n_train, n_test, gamma)
    
    
    lam_list = np.linspace(1e-15,500,CV_grid_n)
    val_err = np.zeros((n_val, len(lam_list)))
    RKHS_dict = {}
    for cv_i, lam in enumerate(lam_list):
        
        alpha_hat = np.linalg.inv(Kernel_Gram+lam*np.eye(n_train))@Y_train
        Y_hat = K_val@alpha_hat
        RKHS_dict[cv_i] = alpha_hat
        val_err[:, cv_i] = Y_val.values-Y_hat
    
    
    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_RKHS = np.mean(np.array(val_err)**2, axis=0)
    
    RKHS_VE_out = {'Model':['RKHS']*2,
                   'Target':[Target]*2,
                'Tune_Param': ['CV_grid','lambda'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        RKHS_VE_out[cn] = [lam_list[i], val_err_RKHS[i]]

    RKHS_VE_out = pd.DataFrame(RKHS_VE_out)
    
    plt.plot(lam_list, val_err_RKHS)
    plt.xlabel('alpha')
    plt.title('Validation Error, RKHS, argmin=%i'%lam_list[min_idx])
    # # plt.show()
    plt.savefig("Figures/RKHS_validation_seed%i.png"%seed)
    plt.close()
    # # plt.show()
    
    # temp = LinearRegression(fit_intercept=True)
    # temp.fit(pd.concat((V_train,W_train_stzd),axis=1),Y_train)
    # Y_hat = temp.predict(pd.concat((V_test,W_test_stzd),axis=1))
    
    Y_hat = K_test@RKHS_dict[min_idx]
    test_err_RKHS = Y_test.values - Y_hat
    RMSE_RKHS = np.sqrt(np.sum(test_err_RKHS**2)/len(test_err_RKHS))

    RKHS_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat,
            'Model': 'RKHS',
            'Seed': seed,
            'Parameter': str(lam_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    RKHS_out = pd.DataFrame.from_dict(RKHS_out)

    #######################################################################################
    ###################################   Neural Net    ###################################
    #######################################################################################
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size = X_train_nnan.shape[0]
    epochs = 20
    n_node_list = np.linspace(1,37,CV_grid_n).astype(int)
    val_err = np.zeros((n_val, len(n_node_list)))
    model_FF_dict = {}
    for cv_i, n_node in enumerate(n_node_list):
        model_FF = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(X_train_nnan.shape[1],)),
                tf.keras.layers.Dense(n_node, activation="relu"),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)
            ]
        )
        # model_FF.summary()
        
        model_FF.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        

        model_FF.fit(X_train_nnan, Y_train, batch_size=batch_size,
                epochs=epochs,verbose=0)
        
        Y_hat = model_FF.predict(X_val_nnan,verbose=0)
        model_FF_dict[cv_i] = model_FF
        val_err[:, cv_i] = Y_val.values - Y_hat.reshape(-1,)
        
    min_idx = np.argmin(np.mean(np.array(val_err)**2, axis=0))
    val_err_NN = np.mean(np.array(val_err)**2, axis=0)

    NN_VE_out = {'Model':['NN']*2,
                 'Target':[Target]*2,
                'Tune_Param': ['CV_grid','n_node'],
                'Seed':[seed]*2,
                'Transformation': Transformation,
                'Window_size': n_train,
                'Validation_size': n_val}
    col_names = ['col_%i'%i for i in range(1,CV_grid_n+1)] 
    for i,cn in enumerate(col_names):
        NN_VE_out[cn] = [n_node_list[i], val_err_NN[i]]

    NN_VE_out = pd.DataFrame(NN_VE_out)

    plt.plot(n_node_list, val_err_NN)
    plt.xlabel('Number of Nodes')
    plt.title('Validation Error, NN, argmin=%i'%n_node_list[min_idx])
    # plt.show()
    plt.savefig("Figures/NN_validation_seed%i.png"%seed)
    plt.close()

    Y_hat = model_FF_dict[min_idx].predict(X_test_nnan,verbose=0)
    test_err_FF = Y_test.values - Y_hat.reshape((-1,))
    RMSE_FF = np.sqrt(np.sum(test_err_FF**2)/len(test_err_FF))

    NN_out = {'Date': Date_used[forecast_idx].dt.strftime("%m/%d/%Y").values,
            'Target': Target,
            'Value': Y_test.values,
            'Prediction': Y_hat.reshape((-1,)),
            'Model': 'NN',
            'Seed': seed,
            'Parameter': str(n_node_list[min_idx]),
            'Window_size': n_train,
            'Validation_size': n_val,
            'Transformation': Transformation
            }
    NN_out = pd.DataFrame.from_dict(NN_out)

    out = np.concatenate((RW_out.values, AR1_out.values, AR12_out.values, AR1_12_out.values,
                           RA_out.values, AR_out.values,RF_out.values, XGB_out.values,
                             XGBs_out.values, Ridge_out.values, Lasso_out.values,
                             AdaLasso_out.values, PCR_out.values, RKHS_out.values, NN_out.values), axis=0)

    query = ''' insert or replace into Results (Date,Target,Value,Prediction,Model,Seed,Parameter,Window_size,Validation_size,Transformation) values (?,?,?,?,?,?,?,?,?,?) '''
    cur.executemany(query, out)
    con.commit()

    
    str_append = ''
    for s in np.arange(1,CV_grid_n+1):
        str_append = str_append+', col_%i'%s

    s = ','.join(['?']*(CV_grid_n+7))
    query = "insert or replace into CV_Error (Model, Target, Tune_Param, Seed, Transformation, Window_size, Validation_size"+str_append+") values ("+ s+ ")"

    out1 = AR_VE_out.values
    out2 = pd.concat((RA_VE_out, RF_VE_out, XGB_VE_out, XGBs_VE_out, Ridge_VE_out, Lasso_VE_out,
                       AdaLasso_VE_out, PCR_VE_out, RKHS_VE_out, NN_VE_out),axis=0).values

    cur.executemany(query, out1)
    con.commit()

    cur.executemany(query, out2)
    con.commit()
    
    cur.close()
    con.close()

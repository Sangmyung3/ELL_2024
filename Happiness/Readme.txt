p1_main_happiness-parameter_tuning.ipynb:		Performs CV to find optimal tuning parameter for fully nonparametric models and store the CV errors in the database.
p2_main_happiness-estimate.ipynb:		Using the parameters found at p1, estimate the nonparametric model and calculate ATE and ATE with contingent variable. Store the results in the database.
p3_main_happiness_get_results.ipynb:		Get the results from the database. Make tables and figures.


p11_main_happiness_PL.ipynb:		Estimate partially linear model with cross fitting. Store the cross validated tuning parameters and estimated betas. Store the results in the database
p21_get_results_PL.ipynb:		Read the results of p11 from the database. Make table.

**p1 should be run before p2. p1 and p2 should be run before p3. p11 should be run before p21.

FINAL DAY: No need to run any programs before running the programs below
p4_main_happiness_PL-NN.ipynb:  Fit neural network and use it to compute ATE and CATE
p31_main_happiness_NN.ipynb:    Use keras Functional API to estimate partially linear model


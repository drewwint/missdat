#!/usr/bin/python3

from missdat.em_fiml import em_fiml

def mcar_test(data, alpha=0.05):
  """
  Performs Todd Little's Missing Completely At Random (MCAR) test using Maximum Likelihood Estimation (MLE).
  
  Code was adapted from the r package "BaylorEdPsych" function LittleMCAR()
    https://rdrr.io/cran/BaylorEdPsych/src/R/LittleMCAR.R
  
  Recomendations:
    We recomend including any variables that can help determine systematic bias
      such as demograpic varaibles or other related varaibles of interst.
  
  Citation for this approach:
    Little, R. J. A. (1988). A test of missing completely at random for 
    multivariate data with missing values. Journal of the American Statistical 
    Association, 83(404), 1198-1202. 
  
  Author
  ------
  Drew E. Winters <drewEwinters@gmail.com>
  
  Parameters:
  -----------
  dat : DataFrame
      A pandas DataFrame containing missing values.
  alpha : float, optional
      Significance level for hypothesis testing (default is 0.05).
  
  Returns:
  --------
  DataFrame:
      A summary of the MCAR test, including:
      - Number of missing patterns
      - Test statistic (X²)
      - Degrees of freedom (df)
      - p-value
      - Interpretation of result
  
  Examples:
  ---------
  Simulating data
    >>> np.random.seed(42)
    >>> data_sim = pd.DataFrame({
         "A": np.random.randn(100),
         "B": np.random.randn(100),
         "C": np.random.randn(100)
     })
   
  Introduce missing values
    >>> data_sim.loc[np.random.choice(100, 10, replace=False), "A"] = np.nan
    >>> data_sim.loc[np.random.choice(100, 15, replace=False), "B"] = np.nan
    >>> data_sim.loc[np.random.choice(100, 20, replace=False), "C"] = np.nan
  
  Examining the ammount of missing values 
  >>> np.sum(np.isnan(data_sim),axis=0)
        A    10
        B    15
        C    20
        dtype: int64
  
  Running test
  >>> mcar_test(data_sim)
        EM algorithm converged at iteration 8
                                                       MCAR Test Values
        number of missing patterns                                    7
        x2                                                     7.016042
        df                                                         17.0
        p                                                       0.98334
        alpha                                                      0.05
        interpretation              Missing Completely at Random (MCAR)
  
  """
  # Require packages import checking
  try:
      import numpy as np
      import pandas as pd
      from scipy.stats import chi2
  except ImportError as e:
      raise ImportError(f"Missing required package: {e.name}. Install it using `pip install {e.name}`")
  
  # Work on a copy to leave the original dataframe unchanged.
  ddt = data.copy()

  # Identify unique missing-data patterns
  missing_patterns = ddt.isnull().astype(int).drop_duplicates()
  pattern_labels = missing_patterns.apply(lambda row: ''.join(row.astype(str)), axis=1)

  # Assign pattern labels to each row in the dataset
  ddt["missing_pattern"] = ddt.isnull().astype(str).agg("".join, axis=1)

  # Count unique missingness patterns
  num_patterns = len(missing_patterns)

  # Impute missing data using MLE (EM FIML algorithm)
  ddt_imputed = em_fiml(ddt.drop(columns=["missing_pattern"]).copy())

  # Compute global mean and covariance (expected under MCAR)
  global_mean = ddt_imputed.mean()
  global_cov = ddt_imputed.cov()

  # Compute Little's MCAR test statistic
  test_stat = 0
  df_total = 0

  for pattern, group in ddt.groupby("missing_pattern"):
      observed_vars = group.drop(columns=["missing_pattern"]).notnull().all()
      observed_data = group.loc[:, observed_vars.index[observed_vars]]
      pattern_size = len(observed_data)
      if pattern_size < 2:
          continue  # Skip if too few observations
      pattern_mean = observed_data.mean()
      # Compute Mahalanobis distance using global parameters for the observed variables
      cols = observed_vars.index[observed_vars]
      mean_diff = pattern_mean - global_mean[cols]
      inv_cov = np.linalg.pinv(global_cov.loc[cols, cols])
      mahalanobis_distance = mean_diff @ inv_cov @ mean_diff.T
      test_stat += pattern_size * mahalanobis_distance
      num_obs_vars = observed_vars.sum()
      df_total += (num_obs_vars * (num_obs_vars + 1)) / 2

  p_value = 1 - chi2.cdf(test_stat, df_total)
  result = "Missing Completely at Random (MCAR)" if p_value >= alpha else "**Not** Random (Potentially MAR or MNAR)"

  miss_df =  pd.DataFrame([
  [num_patterns],
  [test_stat],
  [df_total],
  [p_value],
  [alpha],
  [result]], 
  columns=["MCAR Test Values"],
  index=["number of missing patterns", "x2", "df","p", "alpha","interpretation"])
  
  return miss_df


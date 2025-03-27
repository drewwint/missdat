#!/usr/bin/python3


def em_fiml(data, max_iter=100, tol=1e-4):
  """
  Performs Full Information Maximum Likelihood (FIML) estimation using the EM algorithm.
  
  Author
  ------
  Drew E. Winters <drewEwinters@gmail.com>
  
  Parameters:
  -----------
  data : DataFrame
      A pandas DataFrame containing missing values.
  max_iter : int
      Maximum number of iterations for the EM algorithm.
  tol : float
      Convergence tolerance for stopping the algorithm.
  
  Returns:
  --------
  DataFrame
      The dataset with imputed values (MLE estimates).
  """  
  # Require packages import checking
  try:
      import numpy as np
      import pandas as pd
      from scipy.stats import multivariate_normal
  except ImportError as e:
      raise ImportError(f"Missing required package: {e.name}. Install it using `pip install {e.name}`")

  # Data copying
  data = data.copy()  # Avoid modifying original dataset
  
  # Ensure all columns are numeric
  if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
      raise ValueError("Non-numeric columns detected. Please remove non-numeric columns before using this function.")

  # Initialize imputation
  means = data.mean(skipna=True)
  data_filled = data.fillna(means)
  
  for iteration in range(max_iter):
      old_data = data_filled.copy()
      
      # M-step: update parameter estimates (mean and covariance) using the current filled dataset.
      means = data_filled.mean()
      cov_matrix = data_filled.cov().values  # Covariance matrix as a NumPy array
      
      # E-step: update missing entries based on conditional expectations.
      for idx, row in data.iterrows():
          if row.isnull().any():
              # Determine the observed and missing columns by name.
              observed_cols = row.index[row.notnull()]
              missing_cols = row.index[~row.notnull()]
              
              # Get corresponding positions in the DataFrame (for the covariance matrix)
              obs_idx = [data.columns.get_loc(col) for col in observed_cols]
              mis_idx = [data.columns.get_loc(col) for col in missing_cols]
              
              # Get observed values from the current filled dataset
              obs_vals = data_filled.loc[idx, observed_cols].values
              
              # Partition the mean vector
              mu_obs = means[observed_cols].values
              mu_mis = means[missing_cols].values
              
              # Partition the covariance matrix into sigma_oo and sigma_mo
              sigma_oo = cov_matrix[np.ix_(obs_idx, obs_idx)]
              sigma_mo = cov_matrix[np.ix_(mis_idx, obs_idx)]
              
              # Use pseudo-inverse for numerical stability.
              sigma_oo_inv = np.linalg.pinv(sigma_oo)
              
              # Compute the conditional mean for the missing entries.
              cond_mean = mu_mis + sigma_mo.dot(sigma_oo_inv).dot(obs_vals - mu_obs)
              
              # Update the filled data with the computed conditional mean.
              data_filled.loc[idx, missing_cols] = cond_mean
      
      # Check for convergence: compare the relative change of the filled dataset.
      norm_diff = np.linalg.norm(data_filled.values - old_data.values)
      norm_old  = np.linalg.norm(old_data.values)
      if norm_old > 0 and norm_diff / norm_old < tol:
          break

  return data_filled



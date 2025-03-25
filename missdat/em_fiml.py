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
  cols = data.columns

  # Initialize mean and covariance estimates
  mean = data.mean(skipna=True)
  cov = data.cov()

  for iteration in range(max_iter):
      imputed_data = data.copy()

      # E-step: Estimate missing values using conditional expectation
      for i in range(data.shape[0]):  # Iterate over rows
          missing_mask = data.iloc[i].isnull().to_numpy()  # Convert to NumPy boolean array
          observed_mask = ~missing_mask  # Inverse mask

          if missing_mask.any():  # If there are missing values in this row
              obs_values = data.iloc[i, observed_mask].to_numpy()  # Convert observed values to NumPy array
              obs_mean = mean[observed_mask].to_numpy()
              obs_cov = cov.iloc[observed_mask, observed_mask].to_numpy()

              # Compute conditional expectation E[X_miss | X_obs]
              miss_mean = mean[missing_mask].to_numpy()
              cross_cov = cov.iloc[missing_mask, observed_mask].to_numpy()

              # Compute conditional mean
              if obs_cov.shape[0] > 0:  # Ensure there are observed variables
                  inv_obs_cov = np.linalg.pinv(obs_cov)  # Compute pseudo-inverse
                  cond_mean = miss_mean + cross_cov @ inv_obs_cov @ (obs_values - obs_mean)
              else:
                  cond_mean = miss_mean  # If no observed data, use marginal mean

              # Impute missing values
              imputed_data.iloc[i, missing_mask] = cond_mean

      # M-step: Update parameter estimates
      new_mean = imputed_data.mean()
      new_cov = imputed_data.cov()

      # Check for convergence
      if np.allclose(new_mean, mean, atol=tol) and np.allclose(new_cov, cov, atol=tol):
          print(f"EM algorithm converged at iteration {iteration + 1}")
          break

      # Update estimates for next iteration
      mean, cov = new_mean, new_cov

  return imputed_data


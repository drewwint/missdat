#!/usr/bin/python3

def mcar_test_np(data, num_permutations=1000, random_state=None):
  """
  Non-parametric MCAR test using the association between missingness indicators and row means.
  
  For each column with missing data, this function computes the Spearman correlation between 
  its missing-indicator (1 if missing, 0 if observed) and the overall row mean (computed from all columns, ignoring NaNs).
  The absolute correlations are summed over columns to produce a test statistic.
  
  A permutation procedure is used: for each column, the missing-indicator is randomly permuted,
  and the test statistic is recalculated to form a null distribution. The p-value is then the proportion
  of permuted statistics that exceed or equal the observed statistic.
  
  Parameters
  ----------
  data : pandas.DataFrame
      DataFrame with numeric columns containing missing values.
  num_permutations : int, optional
      Number of permutations to perform (default=1000).
  random_state : int or None, optional
      Seed for reproducibility.
  
  Returns
  -------
  observed_stat : float
      The sum of absolute Spearman correlations between missingness and row means over columns.
  p_value : float
      The p-value from the permutation test.
  
  References
  ----------
  Jamshidian, M., & Jalal, S. (2010). A Comparison of Missing Data Imputation Methods in 
  Longitudinal Studies. Journal of Modern Applied Statistical Methods, 9(2), 353–373.
  """
  # Require packages import checking
  try:
      import numpy as np
      import pandas as pd
      from scipy.stats import spearmanr
  except ImportError as e:
      raise ImportError(f"Missing required package: {e.name}. Install it using `pip install {e.name}`")
  # Copy data to avoid modifying original DataFrame
  data = data.copy()
  rng = np.random.default_rng(random_state)
    
  # Ensure all columns are numeric
  if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
      raise ValueError("Non-numeric columns detected. Please remove non-numeric columns before using this function.")
  
  # Compute an overall row mean using all available data (ignoring NaNs)
  row_means = data.mean(axis=1, skipna=True)
  
  # Compute observed test statistic
  stat_list = []
  for col in data.columns:
    # Skip columns with no missing data
    if data[col].isna().sum() == 0:
        continue
    # Missing indicator: 1 if missing, 0 if observed
    missing_indicator = data[col].isna().astype(int)
    # Use only rows where the row mean is available
    valid = ~row_means.isna()
    if valid.sum() < 5:  # Require at least 5 observations
        continue
    corr, _ = spearmanr(missing_indicator[valid], row_means[valid])
    if np.isnan(corr):
        corr = 0
    stat_list.append(np.abs(corr))
  observed_stat = np.sum(stat_list)
  
  # Permutation: for each column, shuffle the missing indicator and compute test statistic
  perm_stats = []
  for i in range(num_permutations):
    perm_stat_sum = 0
    for col in data.columns:
      if data[col].isna().sum() == 0:
          continue
      missing_indicator = data[col].isna().astype(int)
      # Permute the missing indicator
      permuted_indicator = pd.Series(rng.permutation(missing_indicator.values),
                                     index=missing_indicator.index)
      valid = ~row_means.isna()
      if valid.sum() < 5:
          continue
      corr, _ = spearmanr(permuted_indicator[valid], row_means[valid])
      if np.isnan(corr):
          corr = 0
      perm_stat_sum += np.abs(corr)
    perm_stats.append(perm_stat_sum)
  
  perm_stats = np.array(perm_stats)
  # If no permutation produced a valid statistic, return NaN p-value.
  if perm_stats.size == 0:
    p_value = np.nan
  else:
    p_value = np.mean(perm_stats >= observed_stat)
  
  result = "Missing Completely at Random (MCAR)" if p_value >= 0.05 else "**Not** Random (Potentially MAR or MNAR)"

  miss_df =  pd.DataFrame([
  [observed_stat],
  [p_value],
  [result]], 
  columns=["Non-Parametric MCAR Test Values"],
  index=["t","p","interpretation"])
  
  return miss_df



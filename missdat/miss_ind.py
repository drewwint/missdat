#!/usr/bin/python3

def miss_ind(data):
  """
  Compute a missing indicator for each row in a DataFrame.

  This function checks each row of the input DataFrame to determine if there are any missing (NaN) values.
  It returns a new DataFrame containing a binary indicator:
    - 1 indicates that the row contains at least one missing value.
    - 0 indicates that the row has no missing values.
  
  Including a missing indicator as a control has been shown to improve estimates regardless of imputation approach (in particular machiene learning contexts)
  See: 
    Morvan, M. L., & Varoquaux, G. (2024). Imputation for prediction: beware of diminishing returns. arXiv preprint arXiv:2407.19804.

  Parameters
  ----------
  data : pandas.DataFrame
      A DataFrame containing numeric columns. Non-numeric columns will trigger a ValueError.

  Returns
  -------
  pandas.DataFrame
      A DataFrame with one column named "missing_indicator", where each entry is 1 if the corresponding 
      row in the input has one or more missing values, and 0 if there are no missing values.
  
  """
  # Require packages import checking
  try:
      import numpy as np
      import pandas as pd
  except ImportError as e:
      raise ImportError(f"Missing required package: {e.name}. Install it using `pip install {e.name}`")
  
  # Ensure all columns are numeric
  if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
      raise ValueError("Non-numeric columns detected. Please remove non-numeric columns before using this function.")
  
  # Missing indicator function 
  miss_ind = []
  for ii in range(0,data.shape[0]):
    if sum(np.isnan(data.iloc[ii,:])*1) > 0:
      miss_ind.extend([1])
    else: 
      miss_ind.extend([0])
  
  miss_ind = pd.DataFrame({"missing_indicator": miss_ind})
  
  return miss_ind



#!/usr/bin/python3


def miss_diagnostics(data):
  """
  Conducts missing data diagnostics to assess missingness.
  The output from this function is necessary for assessing missingness and
    reporting missingness features in an academic publication. 
  Such descriptive statistics for missingness is suggested to be reported by:
    - Enders, 2010
    - Janssen et al., 2010 
    - Little & Rubin, 2019
      
  Suggested use for ease of information retrevial:
    miss, cor, all, pat, item = miss_diag(data)
  Then call each dataframe by assigned vector 
  
  Author
  ------
  Drew E. Winters <drewEwinters@gmail.com>
    
  Parameters:
  -----------
  data (DataFrame): A single pandas DataFrame that has missing values.
    
  Returns:
  --------
  missingness_df: Pandas DataFrame
      Binary matrix (1 = missing, 0 = present) for each variable (column) and participant (row).
  missing_corr: Pandas DataFrame
      spearman correlations indicating relationship strength of missing values for missing diagnostics.
      Could loop this into a seaborn heatmap to visualize   e.g.: sns.heaatmap(mcar_diag(data)[1])
  overall_stats: Pandas DataFrame
      statistics for missingness across entire dataframe   including:
          Number of missing patterns
          Proportion of missing data 
          Proportion of complete data
  pattern_stats: Pandas DataFrame
      pattern level diagnostics   that includes:
          N participants missing by pattern
          Number of missing by pattern
          Proportion of missing by pattern
  item_stats: Pandas DataFrame
      item-level statistics to assess and report on missingness at the item-level   including:
          Number of missing values by variable
          Proportion of missing values by variable
          Proportion of complete by variable
  
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
    
  Running function
    >>> miss, cor, all, pat, item = miss_diagnostics(data_sim)
  
  Examining outputs
    >>> miss
             A  B  C
         0   0  0  0
         1   0  0  0
         2   0  0  0
         3   0  1  0
         4   0  0  0
         .. .. .. ..
         95  0  0  0
         96  0  0  0
         97  0  0  0
         98  0  0  0
         99  0  1  0
         
         [100 rows x 3 columns]
       
   >>> cor
                   A         B         C
         A  1.000000  0.046676  0.083333
         B  0.046676  1.000000 -0.140028
         C  0.083333 -0.140028  1.000000
       
   >>> all
                              overall missing stats
         missing_patterns                      7.00
         proportion_missing                    0.15
         proportion_complete                   0.85
       
   >>> pat
                             A  B  C   n
          Missing Pattern 1  0  0  0  61
          Missing Pattern 2  0  0  1  16
          Missing Pattern 3  0  1  0  12
          Missing Pattern 4  0  1  1   1
          Missing Pattern 5  1  0  0   5
          Missing Pattern 6  1  0  1   3
          Missing Pattern 7  1  1  0   2
       
   >>> item
            number_missing  proportion_missing  proportion_complete
         A              10                0.10                 0.90
         B              15                0.15                 0.85
         C              20                0.20                 0.80
          
  """
  # Require packages import checking
  try:
      import numpy as np
      import pandas as pd
  except ImportError as e:
      raise ImportError(f"Missing required package: {e.name}. Install it using `pip install {e.name}`")
  
  # Identifying the number of missing pattens
  num_miss_pat = len(data.isnull().astype(int).drop_duplicates())
  
  # Number of missing values by variable
  number_missing = data.isnull().sum()
    
  # Proportion of missing values by variable
  proportion_missing = data.isnull().mean()
    
  # Average missing across entire dataframe
  proportion_missing_all = np.mean(proportion_missing)
    
  # Proportion of complete cases by variable
  proportion_complete = data.notnull().mean()
    
  # Average complete across entire dataframe
  proportion_complete_all = np.mean(proportion_complete)
  
  # Matrix of missing values: 1 = missing and 0 = not missing
  missingness_df = data.isnull().astype(int)
  
  # Spearman correlation of missingness - indicates relationship strength between missing values
    # note: Spearman is used for dichotomous missingness values
  missing_corr = missingness_df.corr(method= "spearman").loc[missingness_df.corr(method= "spearman").notnull().any(),:]
  
  # Item level missingness
  item_stats = pd.DataFrame({"number_missing":number_missing, 
                             "proportion_missing":proportion_missing,
                             "proportion_complete":proportion_complete})
  
  # Overall missingness
  overall_stats = pd.DataFrame({"missing_patterns":num_miss_pat,
                                "proportion_missing":proportion_missing_all,
                                "proportion_complete":proportion_complete_all},
                                index=["overall missing stats"]).T
  
  # Pattern missingness
  pattern_stats = missingness_df.groupby(list(missingness_df.columns)).size().reset_index(name="n")
  pattern_stats.index = [f"Missing Pattern {i+1}" for i in range(len(pattern_stats))]
  
  return missingness_df, missing_corr, overall_stats, pattern_stats, item_stats







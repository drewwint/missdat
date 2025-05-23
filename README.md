# missdat: a package for missing data diagnostics and estimating missing values

## Overview
`missdat` is a Python package designed to assist in diagnosing and handling missing data within datasets. It provides tools for:

- **Diagnosing Missing Data:** Assess the patterns and extent of missingness.
- **Testing for Missing Completely at Random (MCAR):** Determine if the missing data mechanism is random.
- **Estimating Missing Values:** Impute missing data points using statistical methods.

By integrating `missdat` into your data analysis workflow, you can make informed decisions about handling missing data, leading to more robust statistical inferences.

## What is missdat? 
**missdat** is a Python package that provides useful tools to quickly assess missing data and provide the user with information on how to redress missingness to improve statistical inferences. Specifically, these functions allow the user to perform missing data diagnostics, run a test to determine if data is Missing Completely at Random (MCAR), and estimate the most likely value for that missing data point (currently only maximizing the expected values with Full Information Maximum Likelihood [FIML]). 

## Tutorial
- [Collab Notebook](https://colab.research.google.com/drive/1Dh8Dxf5srIofbsN9KEs1uU1-IAD1pGwV?usp=sharing)

## References
- Enders, C. K. (2010). *Applied Missing Data Analysis*. Guilford Press.
- Janssen, K. J., et al. (2010). Missing covariate data methods for logistic regression: a comparison of approaches in a prediction model context. *Journal of Clinical Epidemiology, 63*(7), 721–729.
- Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data*. Wiley.
- Little, R. J. A. (1988). A test of missing completely at random for multivariate data with missing values. Journal of the American Statistical Association, 83(404), 1198-1202.
- Morvan, M. L., & Varoquaux, G. (2024). Imputation for prediction: beware of diminishing returns. arXiv preprint arXiv:2407.19804.

## Functions
- **miss_diagnostics**
  A function for quantitatively describing patterns of missing data.
  - Procedure:
    * Missing Value Identification:
      The function first identifies the missing values in the dataset and categorizes them into distinct missing patterns based on the number of missing values per participant. 
      This allows for analysis of how different participants have missing data across variables.

    * Pattern-Level Diagnostics:
      The function computes the number of participants that exhibit each missingness pattern and calculates statistics (e.g., number of missing values, proportion missing) for each missing pattern. 
      The dataset is divided into subgroups based on the missingness pattern.

    * Item-Level Diagnostics:
      The function calculates the number of missing values, the proportion of missing values, and the proportion of complete values for each item (variable) in the dataset. 
      These item-level statistics help assess the missingness at a granular level.

     * Spearman Correlation of Missingness:
       A Spearman correlation matrix is computed to measure the relationship between the missingness patterns of the different variables. 
       Spearman correlation is chosen because missingness is treated as a dichotomous variable (missing vs. non-missing).

     * Overall Missingness Statistics:
       The function calculates overall statistics on missingness, including the total proportion of missing data across the entire dataset and the overall proportion of complete data.
 
  This function is particularly useful for:
    - Understanding the structure and patterns of missing data.
    - Reporting missing data diagnostics in academic publications.
    - Visualizing the correlation between missing values in different variables.
    - Assessing missingness at both the item and pattern levels to decide on appropriate imputation strategies

  Reference: Enders (2010)

- **mcar_test**
  A statistical test for randomness in missing data patterns using Little's MCAR test to quantify evidence of MCAR. 
  - Procedure:
     * Missing Data Pattern Identification:
       The function first identifies unique missingness patterns in the dataset 
       by converting missing values to 1 (missing) and non-missing values to 0. 
       It then creates labels for these patterns to distinguish between rows with different missingness profiles.
 
     * FIML Imputation (EM Algorithm):
       The function performs imputation on missing data using the EM (Expectation-Maximization) algorithm. 
       This algorithm uses the all included observed data (full information) to estimate the missing values. 
       It iterates through the following steps:
        -  E-step: Estimate the missing values based on the current mean and covariance of the observed data.
        - M-step: Update the parameters (mean and covariance) based on the newly imputed values.
        - The imputed data is then used for the MCAR test.
 
     * MCAR Test Statistic Calculation:
       The function calculates the Mahalanobis distance for each missingness pattern, 
       which measures how far the observed data deviates from the global mean and covariance 
       (estimated after imputation). The test statistic (X²) is the weighted sum of these distances.
 
     * Degrees of Freedom and p-value:
       The degrees of freedom (df) are computed based on the number of observed variables in each missingness pattern.
       The p-value is then calculated using the Chi-squared distribution.
 
     * Hypothesis Testing:
       The function compares the p-value with the significance level (alpha) to determine whether the data are MCAR. 
       If the p-value is greater than or equal to alpha, the null hypothesis (MCAR) is accepted, meaning that the missingness is deemed random. 
       If the p-value is smaller, the null hypothesis is rejected, indicating that the missingness may not be completely random (i.e., potentially MAR or MNAR).
  
  This function is useful for:
    - Determining if missing data in a dataset is MCAR quantitatively
    - Informing how to redress missing data 

  Reference: Little (1988)

- **mcar_test_np**
  A non-parametric MCAR test using permutation-based approach. 
  - Procedure: 
     * Row Mean Calculation: 
       Computes the mean for each row, ignoring NaN values.

     * Observed Test Statistic Calculation:
       Iterates through each column with missing data.
       Creates a missingness indicator (1 if missing, 0 if observed).
       Computes Spearman’s correlation between the missing indicator and the row mean.
       Takes the absolute value of the correlation and sums it across columns to generate the test statistic.

     * Permutation Testing:
       Randomly permutes the missing indicator for each column and recomputes the test statistic.
       Repeats the process num_permutations times to create a null distribution.

     * Statistical Decision:
       The p-value is calculated as the proportion of permuted test statistics greater than or equal to the observed statistic.
       If p ≥ 0.05, the missingness mechanism is likely MCAR. Otherwise, the data may be Missing At Random (MAR) or Missing Not At Random (MNAR).

  Reference: Janssen et al. (2010)

- **em_fiml**
  A method of imputing values by maximizing the expected values using full information maximum likelihood (FIML).
  - Procedure:
    * Data Preparation:
      Create a copy of the input dataset to preserve the original.
      Extract column names for reference.
      Compute initial estimates of the mean and covariance matrix, ignoring missing values.

    * Full Information Maximum Likelihood (FIML) Imputation Using EM:
      - E-Step: Expectation of Missing Values
        * Iterate through each row of the dataset:
          - Identify missing (`missing_mask`) and observed (`observed_mask`) values.
          - Extract observed values from the current dataset.
          - Partition overall mean vector into observed ('mu_obs') and missing ('mu_mis') parts.
          - Partition the covariance matrix into submatrices:
             * ('sigma_oo'): Covariance matrix among observed variables. 
             * ('sigma_mo'): Covariance between missing and observed variables..
          - Compute the pseudo-inverse of ('sigma_00') for numerical stability.
          - Compute the conditional expectation for the missing values using:
              * E[X_miss | X_obs] = μ_miss + Σ_miss,obs Σ_obs⁻¹ (X_obs - μ_obs)
          - Update the imputed dataset with these conditional expectations.

    * M-Step: Maximization of Parameter Estimates
      -  Re-estimate the mean and covariance matrix from the imputed dataset ('data_filled').

    * Convergence Check and Iteration Control:
      - Compare new estimates to previous ones.
      - If changes in mean and covariance are within tolerance (`tol`), terminate early.
      - Otherwise, continue iterating until `max_iter` is reached. 

  Reference: Little & Rubin (2002)

- **miss_ind**
  Identifying which participants have missing data and returning a dataframe column indicating if there was any missing data. 
  - This can be used as an additional control in analysis for variance in imputation that can be explained by cases that have missing values. 
  
  Reference: Morvan & Varoquaux (2024) 

## Installation

```
# git
git clone https://github.com/drewwint/missdat.git
cd missdat
pip install . 

```

```
# PyPi
pip install missdat

```

### Dependencies
- [NumPy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org)
- [SciPy](https://scipy.org/)

### Usage
``` 
# Simulating data
    >>> np.random.seed(42)
    >>> data_sim = pd.DataFrame({
         "A": np.random.randn(100),
         "B": np.random.randn(100),
         "C": np.random.randn(100)
     })
 
  # Introduce missing values
    >>> data_sim.loc[np.random.choice(100, 10, replace=False), "A"] = np.nan
    >>> data_sim.loc[np.random.choice(100, 15, replace=False), "B"] = np.nan
    >>> data_sim.loc[np.random.choice(100, 20, replace=False), "C"] = np.nan
    
# miss_diagnostics
    >>> miss, cor, all, pat, item = miss_diagnostics(data_sim)
  
  # Examining outputs
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

# mcar_test
   >>> mcar_test(data_sim)
        EM algorithm converged at iteration 8
                                                       MCAR Test Values
        number of missing patterns                                    7
        x2                                                     7.016042
        df                                                         17.0
        p                                                       0.98334
        alpha                                                      0.05
        interpretation              Missing Completely at Random (MCAR)

# non-parametric mcar_test_np
    >>> mcar_test_np(data_sim)
                            Non-Parametric MCAR Test Values
        t                                          0.283454
        p                                             0.323
        interpretation  Missing Completely at Random (MCAR)

# em_fiml
   >>> imp_dat = em_fiml(data_sim)
       EM algorithm converged at iteration 8
   >>> imp_dat
                  A         B         C
       0   0.496714 -1.415371  0.357787
       1  -0.138264 -0.420645  0.560785
       2   0.647689 -0.342715  1.083051
       3   1.523030 -0.549782  1.053802
       4  -0.234153 -0.161286 -1.377669
       ..       ...       ...       ...
       95 -1.463515  0.385317 -0.692910
       96  0.296120 -0.883857  0.899600
       97  0.261055  0.153725  0.307300
       98  0.005113  0.058209  0.812862
       99 -0.234587 -0.023814  0.629629

       [100 rows x 3 columns]

# miss_ind
    >>> miss_ind(data_sim)
            missing_indicator
        0                   0
        1                   0
        2                   0
        3                   1
        4                   0
        ..                ...
        95                  0
        96                  0
        97                  0
        98                  0
        99                  1

        [100 rows x 1 columns]

```





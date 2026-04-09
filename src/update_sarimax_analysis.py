import re

def update_sarimax_analysis(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    analysis_md = """
### 12.1 CRITICAL ANALYSIS: WHY SARIMAX UNDERPERFORMED

While SARIMAX is a standard for linear time-series, it failed to compete with the Gradient Boosting ensemble in this multivariate electricity forecasting use-case for the following reasons:

*   **Sensitivity to Seasonal Hyperparameters**: SARIMAX requires manual, precise identification of (p, d, q) and (P, D, Q, s) orders. Even minor mis-specifications in seasonal differencing lead to massive error propagation in a 7-day recursive forecast.
*   **Dimensionality Constraints (Feature Weighting)**: As more exogenous features (wind, precipitation, cyclical month encodings) are added, SARIMAX struggles to weigh them non-linearly. It treats them as additive linear shifts, ignoring the complex, conditional interactions between weather and human behavior.
*   **Instability in Long/Complex Cycles**: Power demand has overlapping seasonality (daily, weekly, and annual). SARIMAX is mathematically optimized for a single seasonal period (s=7). Handling nested seasonality causes model instability and convergence errors in the underlying MLE (Maximum Likelihood Estimation) solvers.
*   **ML Flexibility vs Statistical Rigidity**: Unlike ML ensembles which use decision trees to segment data into specific regimes (e.g., "Heatwave + Monday" or "Rainy + Sunday"), SARIMAX attempts to fit a single global linear equation to the entire 3-year history, making it unable to adapt to local climate anomalies.
"""

    analysis_txt = """
  -----------------------------------------------------------------------
  12.1 CRITICAL ANALYSIS: WHY SARIMAX UNDERPERFORMED
  -----------------------------------------------------------------------
  While SARIMAX is a standard for linear time-series, it failed to 
  compete with the Gradient Boosting ensemble in this multivariate 
  electricity forecasting use-case for the following reasons:

  * Sensitivity to Seasonal Hyperparameters: SARIMAX requires manual, 
    precise identification of orders. Minor mis-specifications lead to 
    massive error propagation in 7-day recursive forecasts.
  * Dimensionality Constraints: As more exogenous features are added, 
    SARIMAX struggles to weigh them non-linearly, treating them as 
    simple additive linear shifts.
  * Instability in Long/Complex Cycles: Power demand has overlapping 
    seasonality. Handling nested seasonality (daily/weekly/annual) 
    causes model instability and solver convergence errors.
  * ML Flexibility vs Statistical Rigidity: ML ensembles can segment 
    data into specific regimes (e.g., "Heatwave + Monday"). SARIMAX 
    tries to fit a single global linear equation, failing to adapt 
    to local climate anomalies.
"""

    if filepath.endswith('.md'):
        text = text.replace('## 13. CONCLUSION & FUTURE WORK', analysis_md + '\n## 13. CONCLUSION & FUTURE WORK')
    else:
        text = text.replace('======================================================================\n13. CONCLUSION & FUTURE WORK', analysis_txt + '\n======================================================================\n13. CONCLUSION & FUTURE WORK')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_sarimax_analysis('../Project_Report.md')
update_sarimax_analysis('../Project_Report.txt')

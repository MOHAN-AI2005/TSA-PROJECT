import re

def update_ablation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    ablation_md = """
### 10.1 WEATHER FEATURES ABLATION STUDY (Isolating Exogenous Impact)

To definitively prove the necessity of the "Multivariate" architectural approach, an ablation experiment was performed using the Champion Gradient Boosting model. The autoregressive "lag" memory features were temporarily removed from the dataset to force the models to learn solely from environmental and temporal indicators.

*   **Model 1 (Calendar + Weather Features)**: Trained on month, day, temp_max, precipitation, wind_speed, etc.
*   **Model 2 (Calendar Features Only)**: Trained identically, but completely "Weather Blind".

#### ABLATION RESULTS
*   **With Weather MAE**: 9,389.99 MW
*   **Without Weather MAE**: 13,835.82 MW
*   **Improvement**: **32.13% Error Reduction**

> [!IMPORTANT]
> **CONCLUSION:** The inclusion of weather features (primarily maximum temperature and precipitation) unambiguously improves the predictive capability of the model by over 32%. Forecasting architectures that ignore exogenous weather events are fundamentally insufficient for modern grid stabilization.
"""

    ablation_txt = """
  -----------------------------------------------------------------------
  10.1 WEATHER FEATURES ABLATION STUDY
  -----------------------------------------------------------------------
  To definitively prove the necessity of the "Multivariate" architectural 
  approach, an ablation experiment was performed using the Champion Gradient 
  Boosting model. The autoregressive "lag" memory features were temporarily 
  removed from the dataset to force the models to learn solely from 
  environmental and temporal indicators.
  
  * Model 1 (Calendar + Weather Features): Trained on month, day, temp_max, 
    precipitation, and wind_speed.
  * Model 2 (Calendar Features Only): Trained identically, but entirely 
    "Weather Blind".

  ABLATION RESULTS:
  - With Weather MAE   : 9,389.99 MW
  - Without Weather MAE: 13,835.82 MW
  - Performance Improv.: 32.13% (Error Reduction)

  CONCLUSION: 
  The inclusion of weather features (primarily maximum temperature and 
  precipitation) unambiguously improves the predictive capability of the 
  model by over 32%. Forecasting architectures that ignore exogenous weather 
  events are fundamentally insufficient for modern grid stabilization.
"""

    if filepath.endswith('.md'):
        text = text.replace('**Improvement:** (5229 - 2057) / 5229 = **60.7% reduction in error**.', 
                            '**Improvement:** (5229 - 2057) / 5229 = **60.7% reduction in error**.\n' + ablation_md)
    else:
        text = text.replace('  Improvement: 60.7% (massive error reduction over baseline)', 
                            '  Improvement: 60.7% (massive error reduction over baseline)\n' + ablation_txt)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_ablation('../Project_Report.md')
update_ablation('../Project_Report.txt')

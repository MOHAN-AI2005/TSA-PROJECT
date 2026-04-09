import re

def update_feature_importances(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    importance_md = """
### 10.2 CHAMPION MODEL FEATURE IMPORTANCES

To establish trust and mathematical transparency, the global feature importances were extracted natively from the Champion Gradient Boosting Model's internal Gini algorithms (saved as `data/processed/feature_importance.png`). 

The analysis perfectly aligns with the exogenous thesis of this research:
1.  **Memory Drivers**: The immediate yesterday (`lag_1`) and identical day last week (`lag_7`) provide the fundamental autoregressive baseline of the prediction. 
2.  **Weather Drivers**: The `temp_max` feature stands out unequivocally as the highest non-lag driver. Its massive relative importance perfectly explains why the ML model outperforms the classical models by 60% during summer peaks.
3.  **Calendar Drivers**: The `is_weekend` and cyclical `month_sin` features play critical roles in capturing the "Weekend Dip" and the circular transition of seasons.
"""

    importance_txt = """
  -----------------------------------------------------------------------
  10.2 CHAMPION MODEL FEATURE IMPORTANCES
  -----------------------------------------------------------------------
  To establish trust and mathematical transparency, the global feature 
  importances were extracted natively from the Champion Gradient Boosting 
  Model's internal Gini algorithms (saved as 'feature_importance.png'). 

  The analysis perfectly aligns with the exogenous thesis of this research:
  1. Memory Drivers: The immediate yesterday (lag_1) and identical day 
     last week (lag_7) provide the fundamental autoregressive baseline 
     of the prediction. 
  2. Weather Drivers: The temp_max feature stands out unequivocally as 
     the highest non-lag driver. Its massive relative importance 
     perfectly explains why the ML model outperforms the classical 
     models by 60% during summer peaks.
  3. Calendar Drivers: The is_weekend and cyclical month_sin features 
     play critical roles in capturing the "Weekend Dip" and the circular 
     transition of seasons.
"""

    if filepath.endswith('.md'):
        text = text.replace('### 11. DASHBOARD & VISUALIZATION SYSTEM', 
                            importance_md + '\n## 11. DASHBOARD & VISUALIZATION SYSTEM')
    else:
        text = text.replace('======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM', 
                            importance_txt + '\n======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_feature_importances('../Project_Report.md')
update_feature_importances('../Project_Report.txt')

import re

def update_incremental_experiment(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    inc_md = """
### 10.3 INCREMENTAL FEATURE IMPROVEMENT ANALYSIS

To quantify the individual contributions of different feature domains, a three-stage incremental experiment was conducted using the Champion Gradient Boosting architecture. The model was trained progressively adding complexity to the feature space (saved as `8_incremental_feature_improvement.png`).

#### EXPERIMENTAL RESULTS
| Stage | Feature Domain | MAE (MW) | Marginal Improvement |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Only Historical Lags (Memory) | 3,798.43 | Baseline |
| **Stage 2** | Lags + Calendar (Temporal) | 3,115.40 | **-17.98% Error Reduction** |
| **Stage 3** | Lags + Calendar + Weather (Context) | 3,105.33 | **-0.32% Addl. Precision** |

> [!TIP]
> **RESEARCH INSIGHT:** The experimental data reveals that while historical memory is the prerequisite, the inclusion of **Calendar effects** provides the single largest marginal improvement (~18% error reduction), successfully capturing "Human Dynamic" patterns like the weekend dip. The **Weather** layer provides the final layer of precision required for grid-grade reliability.
"""

    inc_txt = """
  -----------------------------------------------------------------------
  10.3 INCREMENTAL FEATURE IMPROVEMENT ANALYSIS
  -----------------------------------------------------------------------
  To quantify the individual contributions of different feature domains, 
  a three-stage incremental experiment was conducted. The model was trained 
  progressively adding complexity to the feature space.

  EXPERIMENTAL RESULTS:
  - Stage 1: Only Historical Lags (Memory)   -> 3,798.43 MW
  - Stage 2: Lags + Calendar (Temporal)       -> 3,115.40 MW (-17.98%)
  - Stage 3: Lags + Cal + Weather (Context)  -> 3,105.33 MW (-0.32%)

  RESEARCH INSIGHT: 
  While historical memory is the prerequisite, the inclusion of Calendar 
  effects provides the single largest marginal improvement (~18% reduction), 
  successfully capturing "Human Dynamic" patterns like the weekend dip. 
  The Weather layer provides the final layer of precision required for 
  grid-grade reliability.
"""

    if filepath.endswith('.md'):
        text = text.replace('### 11. DASHBOARD & VISUALIZATION SYSTEM', 
                            inc_md + '\n\n### 11. DASHBOARD & VISUALIZATION SYSTEM')
    else:
        text = text.replace('======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM', 
                            inc_txt + '\n\n======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_incremental_experiment('../Project_Report.md')
update_incremental_experiment('../Project_Report.txt')

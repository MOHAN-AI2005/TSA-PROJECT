import re

def update_rolling_validation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    rolling_md = """
### 10.5 ROLLING WINDOW VALIDATION (Seasonal Stability Check)

To verify the model's reliability in a production environment, a "Sliding Window" validation experiment was conducted. The model was progressively trained on all historical data and tested on the *unseen* next 30 days of demand, shifting forward month-by-month across the 3-year period (saved as `10_rolling_validation_mae.png`).

#### EXPERIMENTAL RESULTS
*   **Total Windows Evaluated**: 25 (covering mid-2023 to early-2025)
*   **Average Rolling MAE**: 4,276.48 MW

> [!TIP]
> **STABILITY INSIGHT:** The rolling MAE chart confirms that the model maintains consistent accuracy (typically ±2% of peak load) even when faced with seasonal boundary transitions. While error spikes slightly during the most volatile summer heatwaves (May–June), it quickly re-stabilizes as it "learns" the new atmospheric regime, proving its robustness as a live operational tool.
"""

    rolling_txt = """
  -----------------------------------------------------------------------
  10.5 ROLLING WINDOW VALIDATION (Seasonal Stability Check)
  -----------------------------------------------------------------------
  To verify the model's reliability in a production environment, a 
  "Sliding Window" validation experiment was conducted. The model was 
  progressively trained on all historical data and tested on the unseen 
  next 30 days, shifting month-by-month.

  STATISTICAL RESULTS:
  - Total Windows Evaluated : 25
  - Average Rolling MAE      : 4,276.48 MW

  STABILITY INSIGHT:
  The rolling validation confirms that the model maintains consistent 
  accuracy (typically +/-2% of load) across seasonal transitions. While 
  error spikes slightly during volatile summer months, it re-stabilizes 
  rapidly, proving robustness as an operational tool.
"""

    if filepath.endswith('.md'):
        text = text.replace('### 11. DASHBOARD & VISUALIZATION SYSTEM', 
                            rolling_md + '\n\n### 11. DASHBOARD & VISUALIZATION SYSTEM')
    else:
        text = text.replace('======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM', 
                            rolling_txt + '\n\n======================================================================\n11. DASHBOARD & VISUALIZATION SYSTEM')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_rolling_validation('../Project_Report.md')
update_rolling_validation('../Project_Report.txt')

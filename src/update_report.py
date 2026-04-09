import re

def update_report(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Shift TOC numbers for 5 to 14 -> 6 to 15
    for i in range(14, 4, -1):
        if filepath.endswith('.md'):
            text = re.sub(rf'{i}\. \[', f'{i+1}. [', text)
        else:
            text = re.sub(rf'{i}\. ', f'{i+1}. ', text)

    # Shift header numbers
    if filepath.endswith('.md'):
        for i in range(14, 4, -1):
            text = re.sub(rf'## {i}\. ', f'## {i+1}. ', text)
    else:
        for i in range(14, 4, -1):
            text = re.sub(rf'{i}\. ', f'{i+1}. ', text)

    eda_md = """## 5. EXPLORATORY DATA ANALYSIS (EDA)

A comprehensive EDA was performed on the cleaned feature dataset (`data/processed/load_weather_all_features.csv`) to validate the assumptions driving the feature engineering. The analysis generated five professional visualizations stored in `data/eda_outputs/`:

1.  **Time Series Visualization (`1_time_series_load.png`)**:
    Plots the daily maximum demand from 2022 to 2025 alongside a 30-day moving average. This visually confirms the escalating macro-trend of energy consumption over the 3-year period while highlighting sharp, recurring seasonal spikes.

2.  **Monthly Seasonal Trend (`2_monthly_trend.png`)**:
    A bar chart grouping average load by month. This confirms the existence of a definitive annual cycle, peaking massively during summer cooling months and dipping during milder transitional seasons.

3.  **Weekly Pattern & Weekend Dip (`3_weekly_pattern_boxplot.png`)**:
    Boxplots grouped by day-of-week reveal a sharp, consistent drop in load on Saturdays and Sundays. This visualizes the "Weekend Dip" phenomenon caused by a reduction in industrial and commercial grid activity.

4.  **Weather Impact Analysis (`4_weather_impact_scatter.png`)**:
    *   **Temperature Impact**: A scatter plot comparing Maximum Temperature vs Load shows a strong positive correlation, especially as temperatures cross 32°C.
    *   **Precipitation Impact**: A scatter plot comparing Rainfall vs Load shows an inverse effect; high-precipitation days correlate with a marked decrease in grid demand.

5.  **Correlation Heatmap (`5_correlation_heatmap.png`)**:
    A Pearson correlation matrix for all numeric features. This empirically proves the strong correlation between lagged variables and the target load variable.

---

"""
    eda_txt = """======================================================================
5. EXPLORATORY DATA ANALYSIS (EDA)
======================================================================

A comprehensive EDA was performed on the cleaned feature dataset 
(data/processed/load_weather_all_features.csv) to validate the assumptions 
driving the feature engineering. The analysis generated five professional 
visualizations stored in 'data/eda_outputs/':

1. Time Series Visualization (1_time_series_load.png):
   Plots the daily maximum demand from 2022 to 2025 alongside a 30-day 
   moving average. This visually confirms the escalating macro-trend of 
   energy consumption over the 3-year period while highlighting sharp, 
   recurring seasonal spikes.

2. Monthly Seasonal Trend (2_monthly_trend.png):
   A bar chart grouping average load by month. This confirms the existence 
   of a definitive annual cycle, peaking massively during summer cooling 
   months and dipping during milder transitional seasons.

3. Weekly Pattern & Weekend Dip (3_weekly_pattern_boxplot.png):
   Boxplots grouped by day-of-week reveal a sharp, consistent drop in load 
   on Saturdays and Sundays. This visualizes the "Weekend Dip" phenomenon 
   caused by a reduction in industrial and commercial grid activity.

4. Weather Impact Analysis (4_weather_impact_scatter.png):
   * Temperature Impact: A scatter plot comparing Maximum Temperature vs 
     Load shows a strong positive correlation, especially as temperatures 
     cross 32 degrees Celsius.
   * Precipitation Impact: A scatter plot comparing Rainfall vs Load shows 
     an inverse effect; high-precipitation days correlate with a marked 
     decrease in grid demand.

5. Correlation Heatmap (5_correlation_heatmap.png):
   A Pearson correlation matrix for all numeric features. This empirically 
   proves the strong correlation between lagged variables and the target 
   load variable.

"""
    
    if filepath.endswith('.md'):
        # Add to TOC
        text = text.replace('5. [Data Preprocessing Pipeline', '5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)\n6. [Data Preprocessing Pipeline')
        # Insert section before Preprocessing Pipeline
        text = text.replace('## 6. DATA PREPROCESSING PIPELINE', eda_md + '## 6. DATA PREPROCESSING PIPELINE')
    else:
        # Add to TOC 
        text = text.replace('5. Data Preprocessing Pipeline', '5. Exploratory Data Analysis (EDA)\n6. Data Preprocessing Pipeline')
        # Insert section before Preprocessing
        text = text.replace('======================================================================\n6. DATA PREPROCESSING PIPELINE', eda_txt + '\n======================================================================\n6. DATA PREPROCESSING PIPELINE')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_report('../Project_Report.md')
update_report('../Project_Report.txt')

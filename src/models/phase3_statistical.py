import os
import warnings
import pandas as pd
import numpy as np

# We'll import pmdarima and prophet inside the main block to avoid crashing if they aren't fully installed at script startup
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

warnings.filterwarnings('ignore')

def run_models():
    # Attempt imports here
    try:
        import pmdarima as pm
        from prophet import Prophet
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure you have run: pip install pmdarima prophet")
        return

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'load_weather_all_features.csv')

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Exogenous features
    exog_features = ['temp_avg', 'precipitation', 'wind_speed']

    # 80/20 Split
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # ---------------------------------------------------------
    # 2. AUTO ARIMA (NO EXOGENOUS) -> Pure SARIMA
    # ---------------------------------------------------------
    print("\n--- Training SARIMA (No Exogenous Variables) ---")
    print("Running auto_arima (this may take a minute or two)...")
    sarima_model = pm.auto_arima(
        train['load'], 
        seasonal=True, 
        m=7, 
        stepwise=True, 
        suppress_warnings=True, 
        error_action='ignore',
        trace=True
    )
    print("Best SARIMA Order:", sarima_model.order)
    print("Best SARIMA Seasonal Order:", sarima_model.seasonal_order)

    sarima_preds = sarima_model.predict(n_periods=len(test))
    sarima_mae = mean_absolute_error(test['load'], sarima_preds)
    sarima_rmse = root_mean_squared_error(test['load'], sarima_preds)

    # ---------------------------------------------------------
    # 3. AUTO ARIMA (WITH EXOGENOUS) -> SARIMAX
    # ---------------------------------------------------------
    print("\n--- Training SARIMAX (With Weather as Exogenous Variables) ---")
    sarimax_model = pm.auto_arima(
        y=train['load'], 
        X=train[exog_features],
        seasonal=True, 
        m=7, 
        stepwise=True, 
        suppress_warnings=True, 
        error_action='ignore',
        trace=True
    )

    sarimax_preds = sarimax_model.predict(n_periods=len(test), X=test[exog_features])
    sarimax_mae = mean_absolute_error(test['load'], sarimax_preds)
    sarimax_rmse = root_mean_squared_error(test['load'], sarimax_preds)


    # ---------------------------------------------------------
    # 4. PROPHET (BASIC)
    # ---------------------------------------------------------
    print("\n--- Training Prophet (Base Model) ---")
    prophet_train = train[['date', 'load']].rename(columns={'date': 'ds', 'load': 'y'})
    
    m_base = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    m_base.fit(prophet_train)

    future_base = m_base.make_future_dataframe(periods=len(test))
    forecast_base = m_base.predict(future_base)
    prophet_base_preds = forecast_base.iloc[train_size:]['yhat'].values

    prophet_base_mae = mean_absolute_error(test['load'], prophet_base_preds)
    prophet_base_rmse = root_mean_squared_error(test['load'], prophet_base_preds)

    # ---------------------------------------------------------
    # 5. PROPHET (WITH EXOGENOUS WEATHER)
    # ---------------------------------------------------------
    print("\n--- Training Prophet (With Weather Regressors) ---")
    prophet_train_exog = train[['date', 'load'] + exog_features].rename(columns={'date': 'ds', 'load': 'y'})

    m_x = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    for col in exog_features:
        m_x.add_regressor(col)

    m_x.fit(prophet_train_exog)

    future_x = m_x.make_future_dataframe(periods=len(test))
    # Add exogenous variables back manually to future dataframe
    future_x = future_x.merge(df[['date'] + exog_features].rename(columns={'date': 'ds'}), on='ds', how='left')

    forecast_x = m_x.predict(future_x)
    prophet_x_preds = forecast_x.iloc[train_size:]['yhat'].values

    prophet_x_mae = mean_absolute_error(test['load'], prophet_x_preds)
    prophet_x_rmse = root_mean_squared_error(test['load'], prophet_x_preds)

    # ---------------------------------------------------------
    # 6. EVALUATION SUMMARY
    # ---------------------------------------------------------
    print("\n===========================================================")
    print("  PHASE 3 STATISTICAL MODELLING RESULTS")
    print("===========================================================")
    print(f"{'Model Name':<30} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 55)
    print(f"{'SARIMA (Base)':<30} | {sarima_mae:,.0f}      | {sarima_rmse:,.0f}")
    print(f"{'SARIMAX (+ Weather Exog)':<30} | {sarimax_mae:,.0f}      | {sarimax_rmse:,.0f}")
    print(f"{'Prophet (Base)':<30} | {prophet_base_mae:,.0f}      | {prophet_base_rmse:,.0f}")
    print(f"{'Prophet (+ Weather Exog)':<30} | {prophet_x_mae:,.0f}      | {prophet_x_rmse:,.0f}")
    print("===========================================================")
    print("\nNOTE: The original notebook manual SARIMAX scored MAE: ~24,532")

if __name__ == "__main__":
    run_models()

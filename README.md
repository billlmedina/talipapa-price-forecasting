This project began as the foundation for my B.S. in Computer Science thesis, 'Talipapa: An AI-integrated Mobile Application for Real-Time Price Forecasting of Produce Using SARIMA'.

# Agricultural Commodity Price Forecasting

### A Robust Time Series Ensemble Model for Predicting Weekly Agricultural Prices

This project is a complete data-driven pipeline for forecasting the weekly prices of over 40 different agricultural commodities. It addresses the real-world challenges of messy, incomplete time series data by implementing a robust preprocessing and ensemble modeling workflow.

The core of this project is a **weighted ensemble model** that combines the strengths of **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) and **Holt-Winters Exponential Smoothing** to produce more accurate and stable forecasts.

---

## Key Features

* **Ensemble Forecasting:** Combines SARIMA (70%) and Holt-Winters (30%) for superior predictive performance.
* **Robust Data Cleaning:** Implements a multi-stage interpolation (linear, cubic, ffill/bfill) to handle significant gaps in the raw data.
* **Automated Stationarity Handling:** Uses the Augmented Dickey-Fuller (ADF) test to automatically determine the necessary order of differencing (`d`) for each commodity.
* **Dynamic Seasonal Period:** Automatically calculates a relevant seasonal period (`m`) for each time series based on its length.
* **Automated Pipeline:** Processes over 40 commodities in a loop, running the full pipeline from cleaning to forecasting for each one.
* **Clear Outputs:** Generates two clean CSV files: one for future price forecasts and another for the performance metrics (MAE, RMSE, MAPE) of every model.

---

## Data Source
The data used for this project is real-world price data from the Department of Agriculture's Bantay Presyo Program. This dataset serves as the foundation for the predictive model  and contains historical weekly prices for commodities monitored in Philippine wet markets.

Using this operational dataset was a key challenge, as it required a robust preprocessing pipeline to handle:

* Significant missing values (e.g., some commodities had over 50 missing data points).
* Multi-stage interpolation (linear, cubic) to create a viable time series.
* Resampling from a daily to a weekly frequency to stabilize the data for modeling.

---

## Technology Stack

* **Core Libraries:** Python 3.x, pandas, NumPy
* **Time Series Modeling:** statsmodels (SARIMA, Holt-Winters, STL, ADF Test)
* **Model Optimization:** pmdarima (`auto_arima` for hyperparameter tuning)
* **Data Visualization:** Matplotlib & Seaborn
* **Development Environment:** Jupyter Notebook

---

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file can be created with the following key packages:
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels pmdarima jupyter
    ```

4.  **Place Your Data:**
    Ensure your data file, `price_matrix.csv`, is in the root directory of the project.

5.  **Run the Notebook:**
    Open and run all cells in `SarimaEnsemble.ipynb`.

6.  **Check the Outputs:**
    After running, you will find two new files in your directory:
    * `Price_Forecasts.csv`: Contains the 2-week-ahead price forecasts for all commodities.
    * `Model_Performance_Metrics.csv`: Contains the MAE, RMSE, MAPE, and chosen SARIMA orders for each commodity's model.

---

## Project Pipeline: A Step-by-Step Breakdown

This project follows a clear, structured pipeline, which is demonstrated in the notebook.

### 1. Data Health Assessment

Before any processing, the entire `price_matrix.csv` is loaded to visualize the scale of the missing data. A heatmap shows that many commodities (like "Banana (Saba)") have significant gaps, justifying the need for a robust interpolation strategy.



### 2. Data Preprocessing & Cleaning

This is handled by the `preprocess_commodity_data` function for each commodity:

* **Interpolation:** To fill gaps, a multi-stage process is used:
    1.  `interpolate(method='linear')`: Fills small, short-term gaps.
    2.  `interpolate(method='cubic')`: Smooths the curve by filling larger gaps.
    3.  `ffill().bfill()`: A final catch-all to ensure no missing values remain.
* **Resampling:** The raw daily-indexed data is resampled to a weekly frequency (`.resample('W').mean()`) to smooth out daily noise and create a more stable series for modeling.



### 3. Stationarity Analysis

SARIMA models require stationary data (i.e., data where the mean and variance are constant over time).

* **STL Decomposition:** We first use Seasonal-Trend-Loess (STL) decomposition to visually confirm the presence of trends and seasonality that need to be addressed.
* **ADF Test:** The `check_stationarity` function performs an Augmented Dickey-Fuller (ADF) test.
* **Automatic Differencing:** If the test's p-value is above 0.05, the data is non-stationary. The `d` parameter (the order of differencing) is automatically set to `1`, and the model will run on the differenced data.



### 4. Ensemble Model Training & Evaluation

This is the core of the project, handled by the `train_evaluate_ensemble` function.

1.  **Data Split:** The data is split into 80% training and 20% testing sets.

2.  **Model 1: SARIMA:**
    * `pmdarima.auto_arima` is used on the training set to find the optimal non-seasonal (`p,d,q`) and seasonal (`P,D,Q,m`) parameters, minimizing the AIC (Akaike Information Criterion).
    * A full `SARIMAX` model is then fit with these optimized parameters.

3.  **Model 2: Holt-Winters Exponential Smoothing:**
    * An `ExponentialSmoothing` model is also fit to the training data. This model is excellent at capturing trend and seasonal components directly.

4.  **Ensemble & Evaluation:**
    * Predictions are generated from both models on the 20% test set.
    * A final **weighted ensemble prediction** is created:
        `Final Forecast = 0.7 * (SARIMA) + 0.3 * (Holt-Winters)`
    * This ensemble forecast is then scored against the actual test data to generate **MAE**, **RMSE**, and **MAPE** metrics, which are saved to `Model_Performance_Metrics.csv`.

### 5. Final Forecasting & Output

* **Refit on Full Data:** Both the optimized SARIMA and Holt-Winters models are re-trained on 100% of the data.
* **Generate Future Forecast:** The `forecast_future_ensemble` function generates the next 2 weeks of forecasts, again using the 70/30 weighted average.
* **Save Results:** The final forecasts for all commodities are collected and saved to `Price_Forecasts.csv`.

---

## Results

The pipeline generates two clean, high-level summaries.

#### `Price_Forecasts.csv` (Sample Output)

| date       | Banana (Lakatan) | Banana (Latundan) | Banana (Saba) | ... |
| :--------- | :--------------- | :---------------- | :------------ | :-- |
| 2025-06-01 | 106.29           | 74.62             | 50.83         | ... |
| 2025-06-08 | 106.26           | 74.72             | 50.12         | ... |

#### `Model_Performance_Metrics.csv` (Sample Output)

| Commodity         | MAE  | RMSE | MAPE | SARIMA\_Order | Seasonal\_Order    |
| :---------------- | :--- | :--- | :--- | :------------ | :----------------- |
| Banana (Lakatan)  | 4.19 | 4.31 | 3.99 | (1, 0, 0)     | (1, 1, 0, 23)      |
| Banana (Latundan) | 2.36 | 2.49 | 3.19 | (0, 1, 0)     | (0, 1, 0, 23)      |
| Banana (Saba)     | 0.59 | 0.75 | 1.19 | (0, 1, 2)     | (0, 1, 0, 23)      |
| ...               | ...  | ...  | ...  | ...           | ...                |

---

## Future Improvements

* **Dynamic Ensemble Weights:** Tune the 70/30 ensemble weights (e.g., using a grid search or cross-validation) to find the optimal blend for each commodity.
* **Exogenous Variables (SARIMAX):** Incorporate external data (e.g., weather data, fuel prices, transportation costs) as exogenous variables in the SARIMAX model to improve accuracy.
* **Prophet & LSTMs:** Experiment with other models like Facebook's Prophet or deep learning models (LSTMs) and compare their performance.

---

## Author

* **Bill Louis P. Medina**
* www.linkedin.com/in/bill-louis-medina-423381397

import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=DeprecationWarning)

st.set_page_config(
    page_title="Prediction",
    page_icon="📈",
)

st.title("Stock Price Prediction ")

ticker = st.text_input("Enter the stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))
pred_period = st.number_input("Enter the number of days to predict:", min_value=1, value=10)

if st.button("Get Data and Predict"):
    if ticker:
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if not stock_data.empty:
            st.write("Sample data:")
            st.write(stock_data.head())

            stock_data['Date'] = stock_data.index
            stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days
            stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()
            stock_data['EMA_20'] = stock_data['Adj Close'].ewm(span=20, adjust=False).mean()
            stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
            stock_data['High_Low_Range'] = stock_data['High'] - stock_data['Low']
            stock_data = stock_data.dropna()

            st.subheader("Data of Stock")
            st.write(stock_data)

            features = ['Days', 'SMA_20', 'EMA_20', 'Daily_Return', 'High_Low_Range']
            X = stock_data[features].values
            y = stock_data['Adj Close'].values

            train_size = int(len(stock_data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            models = {
                "Linear Regression": LinearRegression(),
                "Polynomial Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            }

            results = []
            best_model_name = None
            best_model = None
            best_mse = float('inf')
            best_r2 = float('-inf')

            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                results.append({
                    "Model": name,
                    "Mean Squared Error (MSE)": mse,
                    "R² Score": r2
                })

                if mse < best_mse:
                    best_mse = mse
                    best_model_name = name
                    best_model = model
                    best_r2 = r2

            results_df = pd.DataFrame(results)
            st.subheader("Models Evaluation Results")
            st.table(results_df)

            st.write(f"Best Model: {best_model_name} with MSE: {best_mse:.2f} and R² Score: {best_r2:.2f}")

            future_days = np.arange(stock_data['Days'].max() + 1, stock_data['Days'].max() + 1 + pred_period).reshape(-1, 1)
            future_sma_20 = stock_data['SMA_20'].iloc[-1]
            future_ema_20 = stock_data['EMA_20'].iloc[-1]
            future_returns = np.zeros(pred_period)
            future_high_low_range = np.zeros(pred_period)

            future_features = np.column_stack([future_days.flatten(), 
                                               np.full(pred_period, future_sma_20), 
                                               np.full(pred_period, future_ema_20), 
                                               future_returns, 
                                               future_high_low_range])

            future_predictions = best_model.predict(future_features)

            predicted_dates = pd.date_range(start=stock_data['Date'].max() + pd.Timedelta(days=1), periods=pred_period)

            plt.figure(figsize=(12, 6))
            plt.plot(stock_data['Date'], stock_data['Adj Close'], label="Historical Prices")
            plt.plot(stock_data['Date'][:train_size], best_model.predict(X_train), label=f"{best_model_name} Predictions on Training Data")
            plt.plot(stock_data['Date'][train_size:], best_model.predict(X_test), label=f"{best_model_name} Predictions on Test Data")
            plt.plot(predicted_dates, future_predictions, label="Future Predictions", linestyle='--')
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title(f"Predicted Prices for {ticker}")
            plt.legend()
            plt.grid(True)

            st.pyplot(plt)

            predicted_df = pd.DataFrame({"Date": predicted_dates, "Predicted Price": future_predictions})
            st.write(predicted_df)
        else:
            st.error("No data found for the selected ticker and date range.")
    else:
        st.info("Please enter a stock ticker to begin.")

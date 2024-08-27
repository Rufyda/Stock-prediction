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

# Set the configuration for the Streamlit app page.
st.set_page_config(
    page_title="Prediction",
    page_icon="📈",
)

# Streamlit app title
st.title("Stock Price Prediction ")

# Input for stock ticker
ticker = st.text_input("Enter the stock ticker (e.g., AAPL, MSFT):", "AAPL")

# Input for date range
start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# Input for prediction period
pred_period = st.number_input("Enter the number of days to predict:", min_value=1, value=10)

# Button to fetch data and make predictions
if st.button("Get Data and Predict"):
    if ticker:
        # Retrieve data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if not stock_data.empty:
            # Prepare data for regression models
            stock_data['Date'] = stock_data.index
            stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

            # Calculate additional features
            stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
            stock_data['EMA_20'] = stock_data['Adj Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average
            stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()  # Daily Returns
            stock_data['High_Low_Range'] = stock_data['High'] - stock_data['Low']  # High-Low Range

            # Drop rows with NaN values due to moving averages
            stock_data = stock_data.dropna()

            # Display the stock data
            st.subheader("Data of Stock")
            st.write(stock_data)

            # Features and target variable
            features = ['Days', 'SMA_20', 'EMA_20', 'Daily_Return', 'High_Low_Range']
            X = stock_data[features].values
            y = stock_data['Adj Close'].values

            # Split the data into training and testing sets
            train_size = int(len(stock_data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Initialize models
            models = {
                "Linear Regression": LinearRegression(),
                "Polynomial Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            }

            # Evaluate models
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
                
                # Store results in a list
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

            # Display the results in a table
            results_df = pd.DataFrame(results)
            st.subheader("Models Evaluation Results")
            st.table(results_df)

            st.write(f"Best Model: {best_model_name} with MSE: {best_mse:.2f} and R² Score: {best_r2:.2f}")

            # Predict future prices using the best model
            future_days = np.arange(stock_data['Days'].max() + 1, stock_data['Days'].max() + 1 + pred_period).reshape(-1, 1)
            future_sma_20 = pd.Series(stock_data['Adj Close']).rolling(window=20).mean().iloc[-1]  # Last SMA value
            future_ema_20 = pd.Series(stock_data['Adj Close']).ewm(span=20, adjust=False).mean().iloc[-1]  # Last EMA value
            future_returns = np.zeros(pred_period)  # Assuming zero returns for simplicity
            future_high_low_range = np.zeros(pred_period)  # Assuming zero range for simplicity
            future_features = np.column_stack([future_days.flatten(), 
                                               np.full(pred_period, future_sma_20), 
                                               np.full(pred_period, future_ema_20), 
                                               future_returns, 
                                               future_high_low_range])

            future_predictions = best_model.predict(future_features)

            # Create the correct dates for the future predictions
            predicted_dates = pd.date_range(start=stock_data['Date'].max() + pd.Timedelta(days=1), periods=pred_period)

            # Plotting the results
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

            # Display the predicted values
            predicted_df = pd.DataFrame({"Date": predicted_dates, "Predicted Price": future_predictions})
            st.write(predicted_df)
        else:
            st.error("No data found for the selected ticker and date range.")
    else:
        st.info("Please enter a stock ticker to begin.")

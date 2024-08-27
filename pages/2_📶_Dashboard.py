import warnings
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from plotly import express as px

st.set_page_config(layout="wide")

# Set the configuration for the Streamlit app page
st.set_page_config(
    page_title="Dashboard",
    page_icon="📶",
)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Dictionary mapping stock tickers to their respective logo image paths
with st.container():

    stocks = {
        'NVDA': 'pages/images/nvda_logo.png',
        'TSLA': 'pages/images/tesla_logo.png',
        'AAPL': 'pages/images/aapl_logo.png',
        'MSFT': 'pages/images/microsoft_logo.png',
        'GC=F': 'pages/images/gold_logo.png',
        'SPY': 'pages/images/spy_logo.png'
    }
    
    # Create a column for each stock and display the logo and name
    cols = st.columns(len(stocks))
    for i, stock in enumerate(stocks):
        with cols[i]:
            st.image(stocks[stock], width=60)
    
    # Display stock ticker names
    stock_names = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GC=F', 'SPY']
    st.markdown("""
    <div style="font-family: monospace;">
        <span style="display:inline-block; width: 115px;"> NVDA</span>
        <span style="display:inline-block; width: 115px;">TSLA</span>
        <span style="display:inline-block; width: 115px;">AAPL</span>
        <span style="display:inline-block; width: 115px;">MSFT</span>
        <span style="display:inline-block; width: 100px;">GC=F</span>
        <span style="display:inline-block; width: 100px;">SPY</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("##")

# Load stock data from a CSV file
data_file_path = 'pages/data/merged_stock_data.csv'
data = pd.read_csv(data_file_path, parse_dates=True, index_col='Date')

# Define columns for stocks and their corresponding colors
required_columns = ['gold', 'tsla', 'msft', 'aapl', 'nvda', 'spy']
colors = ['orange', 'darkred', 'blue', 'gray', 'green', 'skyblue']

st.write("##")

# Plot the adjusted close prices for each stock
st.subheader('Tickers Adj Close Price')
fig = px.line(
    data_frame=data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Adj Close'),
    x='Date',
    y='Adj Close',
    color='Ticker',
    labels={'Adj Close': 'Adjusted Close Price', 'Date': 'Date'},
    color_discrete_map=dict(zip(required_columns, colors))  # Map colors to tickers
)
st.plotly_chart(fig)

# Plot the distribution of stock prices
st.subheader('Distribution of Stocks Prices')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col, color in zip(axes.flatten(), required_columns, colors):
    sns.histplot(data=data, x=col, kde=True, ax=ax, color=color)
    ax.set_title(f'{col.upper()} Distribution')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
st.pyplot(fig)

st.write("##")

# Normalize adjusted close prices for each stock and plot
st.subheader('Tickers Normalizing Adj Close Price')
norm_data = (data / data.iloc[0,:])
fig = px.line(
    data_frame=norm_data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Adj Close'),
    x='Date',
    y='Adj Close',
    color='Ticker',
    labels={'Adj Close': 'Normalized Price', 'Date': 'Date'},
    color_discrete_sequence=['green', 'darkred', 'gray', 'blue', 'orange', 'skyblue']  # Custom color sequence
)
st.plotly_chart(fig)

st.write("##")

# Plot histograms of normalized distributions
st.subheader('Histograms of Normal Distributions')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
columns = required_columns

for ax, col, color in zip(axes.flatten(), columns, colors):
    sns.histplot(data=norm_data, x=col, kde=True, ax=ax, color=color)
    ax.set_title(f'{col.upper()} Normal Distribution')
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.write("##")

# Plot Bollinger Bands for each stock
st.subheader('Bollinger Bands for Stocks')
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Plotting
for ax, col in zip(axes.flatten(), columns):
    rolling_mean = data[col].rolling(window=50).mean()
    rolling_std = data[col].rolling(window=50).std()
    lower_bound = rolling_mean - (2 * rolling_std)
    upper_bound = rolling_mean + (2 * rolling_std)
    
    ax.plot(data.index, data[col], color='blue', label=f'{col.upper()}')
    ax.plot(data.index, rolling_mean, linestyle='--', color='orange', label='Rolling Mean')
    ax.plot(data.index, upper_bound, linestyle='-', color='green', label='Upper Bound')
    ax.plot(data.index, lower_bound, linestyle='-', color='red', label='Lower Bound')
    mean_value = data[col].mean()
    ax.axhline(mean_value, color='black', linestyle='--')
    ax.set_title(f'{col.upper()} Bollinger Bands Indicator')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.grid(alpha=0.10)
    ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.write("##")

# Display daily return statistics
st.subheader('Daily Return Statistcs')
daily_returns = data.pct_change()
daily_returns.iloc[0,:] = 0  # Set the first row to 0 to avoid NaNs
daily_returns = daily_returns.round(2)  # Round the returns for better readability

# Display the statistical summary of daily returns
st.write(daily_returns.describe().round(2).T)

st.write("##")

# Plot daily returns and stock prices
st.subheader('Daily Return of Stocks')
fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

daily_returns['gold'].plot(ax=ax[0][0], color='orange')
ax[0][0].set_title("Gold Daily Returns")

daily_returns['tsla'].plot(ax=ax[0][1], color='darkred')
ax[0][1].set_title("Tesla Daily Returns")

daily_returns['msft'].plot(ax=ax[0][2], color='blue')
ax[0][2].set_title("Microsoft Daily Returns")

data['gold'].plot(kind='line', ax=ax[1][0], color='orange')
ax[1][0].set_title("Gold Price")

data['tsla'].plot(kind='line', ax=ax[1][1], color='darkred')
ax[1][1].set_title("Tesla Price")

data['msft'].plot(kind='line', ax=ax[1][2], color='blue')
ax[1][2].set_title("Microsoft Price")

fig.suptitle("Daily Returns & Stock Price", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)

fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

daily_returns['aapl'].plot(ax=ax[0][0], color='gray')
ax[0][0].set_title("AAPL Daily Returns")

daily_returns['nvda'].plot(ax=ax[0][1], color='green')
ax[0][1].set_title("Nvidia Daily Returns")

daily_returns['spy'].plot(ax=ax[0][2], color='skyblue')
ax[0][2].set_title("SPY Daily Returns")

data['aapl'].plot(kind='line', ax=ax[1][0], color='gray')
ax[1][0].set_title("AAPL Price")

data['nvda'].plot(kind='line', ax=ax[1][1], color='green')
ax[1][1].set_title("NVIDIA Price")

data['spy'].plot(kind='line', ax=ax[1][2], color='skyblue')
ax[1][2].set_title("SPY Price")

fig.suptitle("Daily Returns & Stock Price", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  
st.pyplot(fig)

st.write("##")

# Plot the normal distribution of daily returns
st.subheader('Normal Distribution of Daily Returns')
plt.figure(figsize=(15, 10))
for e, (col, color) in enumerate(zip(required_columns, colors)):
    plt.subplot(3, 3, e + 1)
    sns.histplot(daily_returns[col], kde=True, color=color)
    plt.title("Distribution of " + col)
    plt.tight_layout()
st.pyplot(plt)

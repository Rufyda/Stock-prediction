import warnings
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from plotly import express as px

# Set the configuration for the Streamlit app page
st.set_page_config(
    page_title="Dashboard",
    page_icon="📶", layout="wide"
)

warnings.filterwarnings('ignore', category=DeprecationWarning)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("pages/style/style.css")

# Dictionary mapping stock tickers to their respective logo image paths
stocks = {
    'NVDA': 'pages/images/nvda_logo.png',
    'TSLA': 'pages/images/tesla_logo.png',
    'AAPL': 'pages/images/aapl_logo.png',
    'MSFT': 'pages/images/microsoft_logo.png',
    'GC=F': 'pages/images/gold_logo.png',
    'SPY': 'pages/images/spy_logo.png'
}

# Responsive container for logos
with st.container():
    cols = st.columns([1]*len(stocks))
    for i, stock in enumerate(stocks):
        with cols[i]:
            st.image(stocks[stock], use_column_width=True)  # Use dynamic sizing
            st.markdown(f"<div style='text-align:center;'>{stock}</div>", unsafe_allow_html=True)

# Load stock data from a CSV file
data_file_path = 'pages/data/merged_stock_data.csv'
data = pd.read_csv(data_file_path, parse_dates=True, index_col='Date')

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
    color_discrete_map=dict(zip(required_columns, colors))
)
st.plotly_chart(fig, use_container_width=True)

st.write("##")

# Distribution of stock prices
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
    color_discrete_sequence=['green', 'darkred', 'gray', 'blue', 'orange', 'skyblue']
)
st.plotly_chart(fig, use_container_width=True)

st.write("##")

# Histograms of normalized distributions
st.subheader('Histograms of Normal Distributions')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col, color in zip(axes.flatten(), required_columns, colors):
    sns.histplot(data=norm_data, x=col, kde=True, ax=ax, color=color)
    ax.set_title(f'{col.upper()} Normal Distribution')
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.tight_layout()
st.pyplot(fig)

st.write("##")

# Bollinger Bands for each stock
st.subheader('Bollinger Bands for Stocks')
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

for ax, col in zip(axes.flatten(), required_columns):
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

# Daily return statistics
st.subheader('Daily Return Statistics')
daily_returns = data.pct_change()
daily_returns.iloc[0,:] = 0
daily_returns = daily_returns.round(2)

st.write(daily_returns.describe().round(2).T)

st.write("##")

# Daily returns and stock prices
st.subheader('Daily Return of Stocks')
fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

for i, col in enumerate(['gold', 'tsla', 'msft']):
    daily_returns[col].plot(ax=ax[0][i], color=colors[i])
    ax[0][i].set_title(f"{col.upper()} Daily Returns")
    data[col].plot(kind='line', ax=ax[1][i], color=colors[i])
    ax[1][i].set_title(f"{col.upper()} Price")

fig.suptitle("Daily Returns & Stock Price", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)

fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

for i, col in enumerate(['aapl', 'nvda', 'spy']):
    daily_returns[col].plot(ax=ax[0][i], color=colors[i + 3])
    ax[0][i].set_title(f"{col.upper()} Daily Returns")
    data[col].plot(kind='line', ax=ax[1][i], color=colors[i + 3])
    ax[1][i].set_title(f"{col.upper()} Price")

fig.suptitle("Daily Returns & Stock Price", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)

st.write("##")

# Normal distribution of daily returns
st.subheader('Normal Distribution of Daily Returns')
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

for ax, col, color in zip(axes.flatten(), required_columns, colors):
    sns.histplot(daily_returns[col], kde=True, color=color, ax=ax)
    ax.set_title(f"Distribution of {col.upper()}")
plt.tight_layout()
st.pyplot(fig)

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow import keras
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt 

try: 
    model = load_model('stock_predictions_model.keras')
except Exception as e: 
    st.error(f"Error loading model : {e}")
    st.stop()


st.header('Stock Market Tracker')


# Dropdown for popular stocks
popular_stocks = {
            "Google (GOOG)": "GOOG",
            "Apple (AAPL)": "AAPL",
            "Amazon (AMZN)": "AMZN",
            "Tesla (TSLA)": "TSLA",
            "Microsoft (MSFT)": "MSFT",
            "NVIDIA (NVDA)": "NVDA",
            "Meta (META)": "META",
            "Netflix (NFLX)": "NFLX",
            "Adobe (ADBE)": "ADBE",
            "Intel (INTC)": "INTC",
            "AMD (AMD)": "AMD",
            "IBM (IBM)": "IBM",
            "Oracle (ORCL)": "ORCL",
            "Salesforce (CRM)": "CRM",

            "JPMorgan Chase (JPM)": "JPM",
            "Goldman Sachs (GS)": "GS",
            "Bank of America (BAC)": "BAC",
            "Wells Fargo (WFC)": "WFC",
            "Morgan Stanley (MS)": "MS",
            "Citigroup (C)": "C",
            "American Express (AXP)": "AXP",
            "Visa (V)": "V",
            "Mastercard (MA)": "MA",


            "General Electric (GE)": "GE",
            "Caterpillar (CAT)": "CAT",
            "ExxonMobil (XOM)": "XOM",
            "Chevron (CVX)": "CVX",
            "Boeing (BA)": "BA",
            "3M (MMM)": "MMM",
            "Ford (F)": "F",
            "General Motors (GM)": "GM",


            "Coca-Cola (KO)": "KO",
            "PepsiCo (PEP)": "PEP",
            "Procter & Gamble (PG)": "PG",
            "Walmart (WMT)": "WMT",
            "Nike (NKE)": "NKE",
            "McDonald's (MCD)": "MCD",
            "Starbucks (SBUX)": "SBUX",
            "Costco (COST)": "COST",



            "Johnson & Johnson (JNJ)": "JNJ",
            "Pfizer (PFE)": "PFE",
            "Moderna (MRNA)": "MRNA",
            "Abbott (ABT)": "ABT",
            "Merck (MRK)": "MRK",
            "Gilead Sciences (GILD)": "GILD",


            "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
            "Tata Consultancy Services (TCS.NS)": "TCS.NS",
            "Infosys (INFY.NS)": "INFY.NS",
            "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
            "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
            "State Bank of India (SBIN.NS)": "SBIN.NS",
            "Axis Bank (AXISBANK.NS)": "AXISBANK.NS",
            "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
            "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
            "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS",
            "Adani Ports (ADANIPORTS.NS)": "ADANIPORTS.NS",
            "Maruti Suzuki (MARUTI.NS)": "MARUTI.NS",
            "ITC Limited (ITC.NS)": "ITC.NS",
            "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
            "Larsen & Toubro (LT.NS)": "LT.NS",
            "Wipro (WIPRO.NS)": "WIPRO.NS",
            "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
            "Tata Steel (TATASTEEL.NS)": "TATASTEEL.NS",
            "Power Grid (POWERGRID.NS)": "POWERGRID.NS"
}

selected_stock = st.selectbox("Select a popular stock : ", list(popular_stocks.keys()))

custom_stock = st.text_input("Or enter a custom stock symbol (e.g., GOOG)", "")

stock = custom_stock if custom_stock else popular_stocks[selected_stock]


start = '2012-01-01'
end = '2024-12-31'

try: 
    data = yf.download(stock, start, end)
    if data.empty: 
        st.warning("No data found for that stock symbol, try another one.")
        st.stop()
except Exception as e: 
    st.error(f"Failed to fetch data : {e}")
    st.stop()


st.subheader('stock data')
st.write(data)


data_train = pd.DataFrame(data.Close[0 : int(len(data) * 0.8)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8) : len(data)])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

data_train_scaler = scaler.fit_transform(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)


data_test_scaled = scaler.transform(data_test)


st.subheader('Price vs MA 50')

ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)


st.subheader('Price vs MA 50 vs MA 100')

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.plot(ma_100_days, 'b')
st.pyplot(fig2)


st.subheader('Price vs MA 100 vs MA 200')

ma_200_days = data.Close.rolling(200).mean()
ma_100_days = data.Close.rolling(100).mean()
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.plot(ma_100_days, 'b')
st.pyplot(fig3)



x = [] 
y = [] 

for i in range(100, data_test_scaled.shape[0]): 
    x.append(data_test_scaled[i - 100 : i])
    y.append(data_test_scaled[i, 0])


x, y = np.array(x), np.array(y)

prediction = model.predict(x)

scale = 1 / scaler.scale_

prediction = prediction * scale

y = y * scale


st.subheader(f"{stock} Price vs {stock} Predicted Price")

fig4 = plt.figure(figsize=(10, 8))
plt.plot(y, 'r', label='actual price')
plt.plot(prediction, 'g', label='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig4)


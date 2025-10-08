# StockSense

# launch the app via
  https://stock-price-prediction-kvqrft36f2attw539dbgbj.streamlit.app/

# 📈 StockSense — AI Stock Price Predictor

A Streamlit-powered web app that predicts stock prices using an LSTM deep learning model.  
It visualizes real-time trends, moving averages, and compares actual vs predicted values.

---

## 🚀 Features

- 🧠 Deep Learning Model (LSTM) trained on 10+ years of data  
- 💹 Real-time price fetching via [Yahoo Finance](https://finance.yahoo.com/)  
- 📊 Interactive moving average charts (MA50, MA100, MA200)  
- 🔮 Predicts next-day closing price  
- 🌈 Clean, modern Streamlit UI  

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|----------|
| Python | Core language |
| Streamlit | Web App Framework |
| Keras / TensorFlow | Deep Learning Model |
| Pandas, NumPy | Data Processing |
| Matplotlib | Visualization |
| yFinance | Stock Data API |

---

## 📦 Installation

```bash
git clone https://github.com/tarupathak30/stock-price-prediction.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
streamlit run app.py

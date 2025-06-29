# CrudeOil-Predictor-XGBoost

**WTI-Forecaster-XDays** is a machine learning project designed to predict the future price of WTI crude oil (West Texas Intermediate) *X days ahead*, optimized for 1D or 7D, using machine learning models boosted by XGBoost and economic, financial, and geopolitical data.

---

## 🚀 Main Features

- 🔮 Future price prediction of WTI with different time horizons (`X` days).
  
- 📊 Integration of multiple data sources:
  
  - Historical prices of WTI  
  - S&P 500 and US Dollar  
  - GPR (Geopolitical Risk Index)  
  - Maritime traffic through Suez and Panama canals  
  - Stock prices of the oil sector (Exxon Mobil)  
    
- ⚙️ Training with GPU-accelerated `XGBoost`.
  
- 🧠 Generation of advanced features: momentum, volatility, extreme events, moving averages, etc.
  
- 📈 Visualization of results and performance evaluation with key metrics.

---

## 📁 Project Structure

Modelo Koala/  
  ├── Codigo/  
  │ ├──Datos/  
  │ │  ├── Crude Oil WTI Futures Historical Data expandido.csv  
  │ │  ├── Crude Oil WTI Futures Historical Data expandido2.csv  
  │ │  ├── Crude Oil WTI Futures Historical Data Predecir.csv  
  │ │  ├── datos_mensuales_limpio.csv  
  │ │  ├── DollarPredecir27_06.csv   
  │ │  ├── Exxon Mobil Stock Price Expandido.csv  
  │ │  ├── Exxon Mobil Stock Price Predecir.csv  
  │ │  ├── Gpr_por_Dia.csv  
  │ │  ├── GPR_De_paises_por_dia.csv  
  │ │  ├── S&P 500 Futures Historical Data expandido.csv  
  │ │  ├── trafico_canales1986_2025.csv  
  │ │  ├── S&P 500 Futures Historical Data Predecir.csv  
  │ │  ├── US Dollar Index Historical Extendido.csv  
  │ │  ├── US Dollar Index Historical Predecir.csv  
  │ │  ├──  WTIPredecir27_06.csv  
  │ │  ├── S&P 500Predecir27_06.csv    
  │ │  └── README.md  
  │ ├── Panda11/  
  │ │  ├── Panda11.py    
  │ │  ├── predecir11.py  
  │ │  ├── Parametros.py  
  │ │  └── README.md  
  └── README.md  

---

## Results

### Model trained for 1 day:
#### Training at 1 day:
![Training1](/images/Train1.png)  
![Training2](/images/Train2.png)  
![Training3](/images/Train3.png)  

### Prediction at 1 Day
![Prediction1](/images/Predic1.png)  

- 📊 MODEL METRICS (Close):  
  - MAE:  0.9680  
  - RMSE: 1.2303  
  - R²:   0.9537  
  - MAPE: 2.38%  

---

### Model trained for 7 days:
#### Training at 7 days:
![Training4](/images/Train4.png)  
![Training5](/images/Train5.png)  
![Training6](/images/Train6.png)  

### Prediction at 7 Days
![Prediction2](/images/Predic2.png)  

- 📊 MODEL METRICS (Close):  
  - MAE:  1.9140  
  - RMSE: 2.4920  
  - R²:   0.8065  
  - MAPE: 4.50%  



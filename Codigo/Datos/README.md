# CrudeOil-Predictor-XGBoost

## Data
##### In this project, I progressively used various data sources to train the model. Initially, I used only the historical WTI Crude Oil prices and the GPR, but gradually added and removed variables that appeared to correlate with the WTI price in an attempt to improve the model.
---

## 🚀 Main Inputs

- Historical WTI Price:
  - Direct variables:
    - Open
    - Low
    - High
    - Vol.
    - Change
      
  - Indirect variables:
    - Pricelag1
    - Price_lag2
    - Pendiente
    - Alto-Bajo
    - Bajo-Alto
    - Price_lag3
    - media_col1
    - desv_col1
    - skew_col1
    - kurt_col1
    - cambio_pct
    - evento_extremo
    - Vol_WTI_extrema
      
- Historical S&P500 Price:
  - Direct variables:
    - N/A
        
  - Indirect variables:
    - media_sp
    - desv_sp
    - skew_sp
    - kurt_sp
    - momentum_sp

- Historical Dollar Index:
  - Direct variables:
    - ChangeDolar
      
  - Indirect variables:
    - momentum_dolar
    - media_Dolar
    - desv_Dolar
    - skew_Dolar
      
- Ships through Panama and Suez:
  - Direct variables:
    - barcos_panamá
    - barcos_suez
  
  - Indirect variables:
    - N/A

- Historical Price of Exxon Mobil Stock:
  - Direct variables:
    - N/A
      
  - Indirect variables:
    - momentum_EP

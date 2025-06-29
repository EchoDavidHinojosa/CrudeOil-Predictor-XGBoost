# CrudeOil-Predictor-XGBoost

## Datos
##### En este proyecto he ido utilizando diversos datos para entrenar al modelo,inicialmente utilicé únicamente un histórico del WTI crude oil y el gpr,aunque poco a poco fuí introduciendo nuevos y eliminando datos que parecían tener cierta correlación con el precio del WTI crude oil con la  intención de mejorar el modelo.
---

## 🚀 Input principales

- Precio histórico del WTI:
  - Variables directas
    - Open
    - Low
    - High
    - Vol.
    - Change
      
  - Variables indirectas
    
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
      
- Precio histórico del S&P500
  
  -  Variables directas
      - N/A
        
  - Variables indirectas
    - media_sp
    - desv_sp
    - skew_sp
    - kurt_sp
    - momentum_sp
- Precio histórico del dolar
  
  - Variables directas
    - ChangeDolar
      
  - Variables indirectas
    - momentum_dolar
    - media_Dolar
    - desv_Dolar
    - skew_Dolar
      
- Barcos por Panamá y Suez
  - Variables directas
    - barcos_panamá
    - barcos_suez
  - Variables indirectas
      - N/A
- Predio Histórico de Exxon Mobil Stock
  
  - Variables directas
    - N/A
      
  - Variables indirectas
    - momentum_EP
      

---



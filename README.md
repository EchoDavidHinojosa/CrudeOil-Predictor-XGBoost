# CrudeOil-Predictor-XGBoost

**WTI-Forecaster-XDays** es un proyecto de machine learning diseñado para predecir el precio futuro del petróleo WTI (West Texas Intermediate) a *X días vista*, utilizando modelos de aprendizaje automático optimizados con XGBoost y datos económicos, financieros y geopolíticos.

---

## 🚀 Características principales

- 🔮 Predicción del precio futuro del WTI con diferentes horizontes de tiempo (`X` días).
- 📊 Integración de múltiples fuentes de datos:
  - Precios históricos del WTI
  - S&P 500 y Dólar estadounidense
  - GPR (Geopolitical Risk Index)
  - Tráfico en canales marítimos (Suez y Panamá)
  - Precio de acciones del sector petrolero (Exxon Mobil)
- ⚙️ Entrenamiento con `XGBoost` acelerado por GPU.
- 🧠 Generación de variables avanzadas: momentum, volatilidad, eventos extremos, medias móviles, etc.
- 📈 Visualización de resultados y evaluación del rendimiento con métricas clave.

---

## 📁 Estructura del proyecto

Modelo Koala/  
├── Datos/  
│   ├── Crude Oil WTI Futures Historical Data expandido.csv  
│   ├── Gpr_por_Dia.csv  
│   ├── GPR_De_paises_por_dia.csv  
│   ├── trafico_canales1986_2025.csv  
│   ├── S&P 500 Futures Historical Data expandido.csv  
│   ├── US Dollar Index Historical Extendido.csv  
│   └── Exxon Mobil Stock Price Expandido.csv  
├── modelo_koala.py  
├── Parametros.py  
├── requirements.txt  
└── README.md  



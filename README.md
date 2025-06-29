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
WTI-Forecaster-XDays/
├── main.py # Script principal de procesamiento, entrenamiento y evaluación
├── Parametros.py # Archivo de configuración (nombre del modelo, días de predicción)
├── Datos/ # Archivos de entrada (.csv)
├── Modelos/ # Carpeta sugerida para guardar los modelos entrenados
├── README.md # Documentación del proyecto
└── requirements.txt # Requisitos del entorno

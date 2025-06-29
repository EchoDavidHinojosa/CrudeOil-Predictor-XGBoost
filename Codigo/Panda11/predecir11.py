import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import Parametros
from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


print("🚀 Cargando datos nuevos para predicción...")

# Cargar datos
precio_nuevo = pd.read_csv('../Datos/Crude Oil WTI Futures Historical Data predecir.csv', parse_dates=['fecha'])
gpr_nuevo = pd.read_csv('../Datos/Gpr_por_Dia.csv', parse_dates=['fecha'])
mensual = pd.read_csv('../Datos/GPR_De_paises_por_dia.csv', parse_dates=['Date'])
sp500=pd.read_csv('../Datos/S&P 500 Futures Historical Data Predecir.csv', parse_dates=['Date'])
sp500.rename(columns={'Date': 'fecha', 'Price': 'precioSp', 'Open': 'OpenSp','High':'HighSp','Low':'LowSp','Vol.':'Vol.SP','Change':'ChangeSP'}, inplace=True)
sp500['fecha'] = sp500['fecha'].dt.normalize()

dolar = pd.read_csv('../Datos/US Dollar Index Historical Predecir.csv', parse_dates=['fecha'])

dolar.rename(columns={'Date': 'fecha', 'Price': 'precioDolar', 'Open': 'OpenDolar','High':'HighDolar','Low':'LowDolar','Vol.':'Vol.Dolar','Change':'ChangeDolar'}, inplace=True)
dolar['fecha'] = dolar['fecha'].dt.normalize()

empresa_Petroleo = pd.read_csv('../Datos/Exxon Mobil Stock Price Expandido.csv', parse_dates=['fecha'])
empresa_Petroleo['fecha'] = empresa_Petroleo['fecha'].dt.normalize()
empresa_Petroleo.rename(columns={'Date': 'fecha', 'Price': 'PriceEP', 'Open': 'OpenEP','High':'HighEP','Low':'LowEP','Vol.':'Vol.EP','Change':'ChangeEP'}, inplace=True)





# Cargar y preparar datos de barcos (CSV mensual)
barcos = pd.read_csv('../Datos/trafico_canales1986_2025.csv', parse_dates=['Date'])
barcos.rename(columns={'Date': 'fecha','Panama': 'barcos_panamá','Suez': 'barcos_suez'}, inplace=True)
barcos['fecha'] = barcos['fecha'].dt.normalize()
mensual.rename(columns={'Date': 'fecha'},inplace=True)
# Limpiar y convertir Open y Vol
precio_nuevo['Open'] = pd.to_numeric(precio_nuevo['Open'], errors='coerce')
precio_nuevo['Vol.'] = precio_nuevo['Vol.'].replace({',': ''}, regex=True)
precio_nuevo['Vol.'] = pd.to_numeric(precio_nuevo['Vol.'], errors='coerce')

# Normalizar fechas
precio_nuevo['fecha'] = precio_nuevo['fecha'].dt.normalize()
gpr_nuevo['fecha'] = gpr_nuevo['fecha'].dt.normalize()
mensual['fecha'] = mensual['fecha'].dt.normalize()

# Expandir mensual a diario para mensual y barcos
fechas_diarias = pd.DataFrame({'fecha': pd.date_range(start=precio_nuevo['fecha'].min(), 
                                                      end=precio_nuevo['fecha'].max(), freq='D')})

mensual_diario = fechas_diarias.merge(mensual, on='fecha', how='left').ffill()
barcos_diario = fechas_diarias.merge(barcos, on='fecha', how='left').ffill()
Sp500_diario= fechas_diarias.merge(sp500, on='fecha', how='left').ffill().infer_objects()
dolar_diario=fechas_diarias.merge(dolar, on='fecha', how='left').ffill().infer_objects()
EP_diario=fechas_diarias.merge(empresa_Petroleo, on='fecha', how='left').ffill().infer_objects()
# Merge completo
df_nuevo = precio_nuevo.merge(gpr_nuevo, on='fecha').merge(mensual_diario, on='fecha', how='left')
df_nuevo = df_nuevo.merge(barcos_diario, on='fecha', how='left').sort_values('fecha')
df_nuevo = df_nuevo.merge(Sp500_diario, on='fecha', how='left')
df_nuevo = df_nuevo.merge(dolar_diario, on='fecha', how='left')
df_nuevo= df_nuevo.merge(EP_diario, on='fecha', how='left')



print(f"✅ Filas después del merge completo: {df_nuevo.shape[0]}")

# Target y fecha objetivo
df_nuevo['precio_objetivo'] = df_nuevo['Price'].shift(-Parametros.dias)
df_nuevo['fecha_objetivo'] = df_nuevo['fecha'].shift(-Parametros.dias)
df_nuevo['Price_lag1'] = df_nuevo['Price'].shift(1)
df_nuevo['Price_lag2'] = df_nuevo['Price'].shift(1)-df_nuevo['Open'].shift(1)
df_nuevo['Pendiente']=df_nuevo['Price'].shift(2)-df_nuevo['Price'].shift(1)
df_nuevo['Alto-Bajo']=df_nuevo['High'].shift(1)-df_nuevo['Low'].shift(1)
df_nuevo['Bajo-Alto']=df_nuevo['Low'].shift(1)-df_nuevo['High'].shift(1)
df_nuevo['Price_lag3'] = df_nuevo['Price']-df_nuevo['Price'].shift(1)
df_nuevo['media_col1'] = df_nuevo['Price'].mean()
df_nuevo['desv_col1'] = df_nuevo['Price'].std()
df_nuevo['skew_col1'] = df_nuevo['Price'].skew()
df_nuevo['kurt_col1'] = df_nuevo['Price'].kurt()
df_nuevo['cambio_pct'] = df_nuevo['Price'].pct_change()
df_nuevo['evento_extremo'] = (df_nuevo['cambio_pct'].abs() > 0.05).astype(int)
df_nuevo['PrecioEnteroSP']=df_nuevo['precioSp']*1000+df_nuevo['PriceD']
#df_nuevo['OpenEnteroSP']=df_nuevo['OpenSp']*1000+df_nuevo['OpenD']
#df_nuevo['HighEnteroSP']=df_nuevo['HighSp']*1000+df_nuevo['HighD']
#df_nuevo['LowEnteroSP']=df_nuevo['LowSp']*1000+df_nuevo['LowD']
df_nuevo['media_sp'] = df_nuevo['PrecioEnteroSP'].mean()
df_nuevo['desv_sp'] = df_nuevo['PrecioEnteroSP'].std()
df_nuevo['skew_sp'] = df_nuevo['PrecioEnteroSP'].skew()
df_nuevo['kurt_sp'] = df_nuevo['PrecioEnteroSP'].kurt()
# Calculamos el momentum a 10 días
df_nuevo['momentum_sp'] = df_nuevo['PrecioEnteroSP'] - df_nuevo['PrecioEnteroSP'].shift(30)
df_nuevo['GPRDCambio'] = df_nuevo['GPRD'].pct_change()
df_nuevo['GPRD_extremo'] = (df_nuevo['GPRDCambio'].abs() > 0.05).astype(int)
df_nuevo['momentum_dolar'] = df_nuevo['precioDolar'] - df_nuevo['precioDolar'].shift(10)
df_nuevo['momentum_EP'] = df_nuevo['PriceEP'] - df_nuevo['PriceEP'].shift(4)
df_nuevo = df_nuevo.dropna(subset=['precio_objetivo'])
# Calcular GPRD_Delta
df_nuevo['GPRD'] = pd.to_numeric(df_nuevo['GPRD'], errors='coerce')
df_nuevo['GPRD_Delta'] = df_nuevo['GPRD'] - df_nuevo['GPRD'].shift(3)


















# Conversión columnas GPR
cols_obj = ['GPRD','GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA30', 'GPRD_MA7']
for col in cols_obj:
    if col in df_nuevo.columns:
        try:
            df_nuevo[col] = pd.to_numeric(df_nuevo[col], errors='raise')
        except:
            df_nuevo[col] = df_nuevo[col].astype('category')

# Conversión columnas mensuales
columnas_mensuales = ['GPRC_EGY', 'GPRC_ISR', 'GPRC_SAU', 'GPRC_USA']
for col in columnas_mensuales:
    if col in df_nuevo.columns:
        df_nuevo[col] = pd.to_numeric(df_nuevo[col], errors='coerce')

# Conversión columnas barcos a numérico
for col in ['barcos_suez', 'barcos_panamá']:
    if col in df_nuevo.columns:
        df_nuevo[col] = pd.to_numeric(df_nuevo[col], errors='coerce')

# Conversión de object restantes
for col in df_nuevo.select_dtypes(include='object').columns:
    try:
        df_nuevo[col] = pd.to_numeric(df_nuevo[col], errors='raise')
        print(f"✅ '{col}' convertido a numérico.")
    except:
        df_nuevo[col] = df_nuevo[col].astype('category')
        print(f"ℹ️ '{col}' convertido a categoría.")

# --- Crear Price_Media_Mes_Pasado ---
df_nuevo['mes'] = df_nuevo['fecha'].dt.to_period('M')
promedios_mensuales = df_nuevo.groupby('mes')['Price'].mean().rename('Price_Media_Mes')
df_nuevo = df_nuevo.join(promedios_mensuales, on='mes')
df_nuevo['Price_Media_Mes_Pasado'] = df_nuevo['Price_Media_Mes'].shift(1)
df_nuevo['Price_Media_Mes_Pasado'] = df_nuevo['Price_Media_Mes_Pasado'].bfill()
df_nuevo['Vol_WTI'] = df_nuevo['Open'].pct_change().rolling(window=8).std()
umbral_vol_wti = df_nuevo['Vol_WTI'].quantile(0.97)
df_nuevo['Vol_WTI_extrema'] = (df_nuevo['Vol_WTI'] > umbral_vol_wti).astype(int)
#Experimentos 

df_nuevo['Vol_SP'] = df_nuevo['PrecioEnteroSP'].pct_change().rolling(window=5).std()
umbral_vol_sp = df_nuevo['Vol_SP'].quantile(0.9)
df_nuevo['Vol_SP_extrema'] = (df_nuevo['Vol_SP'] > umbral_vol_sp).astype(int)




# Excluir columnas irrelevantes y target 'Price' de features
X_nuevo = df_nuevo.drop(columns=['fecha','PrecioEnteroSP', 'Vol.EP','GPRDCambio','precio_objetivo','precioDolar', 'HighDolar','LowDolar','OpenDolar','Price','OpenD','PriceD','precioSp','OpenSp','HighSp','HighD','LowSp','LowD',
                     'Vol.SP','PriceEP','OpenEP','HighEP','ChangeEP','LowEP','DAY','Vol_WTI','Vol_SP','ChangeDolar'])

print(f"📌 Features usadas en la predicción: {list(X_nuevo.columns)}")

# Cargar modelo
modelo = xgb.Booster()
modelo.load_model(Parametros.nombre + str(Parametros.dias) + "Dias.json")
# Orden correcto esperado por el modelo
orden_columnas_modelo = [
    'Open', 'High', 'Low', 'Vol.', 'N10D', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT',
    'GPRD_MA30', 'GPRD_MA7', 'event', 'GPRC_CHN', 'GPRC_EGY', 'GPRC_ISR', 'GPRC_RUS',
    'GPRC_SAU', 'GPRC_USA', 'GPRC_VEN', 'barcos_panamá', 'barcos_suez', 'Change ',
     'GPRD_Delta',
    'Price_Media_Mes_Pasado', 'Price_lag1', 'Price_lag2', 'Pendiente', 'Alto-Bajo',
    'Bajo-Alto', 'Price_lag3', 'media_col1', 'desv_col1', 'skew_col1', 'kurt_col1',
    'cambio_pct', 'evento_extremo', 'media_sp', 'desv_sp', 'skew_sp', 'kurt_sp',
    'GPRD_extremo','momentum_EP', 'momentum_sp','momentum_dolar','Vol_WTI_extrema','Vol_SP_extrema'
]
 
#['Open', 'High', 'Low', 'Vol.', 'DAY', 'N10D', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA30', 'GPRD_MA7', 'event', 'GPRC_CHN', 'GPRC_EGY', 'GPRC_ISR', 'GPRC_RUS', 'GPRC_SAU', 'GPRC_USA', 'GPRC_VEN', 'barcos_panamá', 'barcos_suez', 'precioSp', 'PriceD', 'OpenSp', 'OpenD', 'HighSp', 'HighD', 'LowSp', 'LowD', 'Vol.SP', 'Change ', 'GPRD_Delta', 'Price_Media_Mes_Pasado', 'Price_lag1', 'Price_lag2', 'Pendiente', 'Alto-Bajo', 'Bajo-Alto', 'Price_lag3', 'media_col1', 'desv_col1', 'skew_col1', 'kurt_col1', 'cambio_pct', 'evento_extremo', 'PrecioEnteroSP', 'OpenEnteroSP', 'HighEnteroSP', 'LowEnteroSP'] 




X_nuevo = X_nuevo[orden_columnas_modelo]

# Crear DMatrix para predicción
dnew = xgb.DMatrix(X_nuevo, enable_categorical=True)

# Predicción
best_iter = getattr(modelo, 'best_iteration', None)
if best_iter is not None:
    y_pred = modelo.predict(dnew, iteration_range=(0, best_iter + 1))
else:
    y_pred = modelo.predict(dnew)

df_nuevo['prediccion'] = y_pred

# Filtrar resultados válidos (donde tenemos precio objetivo real)
df_resultados = df_nuevo.dropna(subset=['precio_objetivo'])

# Métricas
y_true = df_resultados['precio_objetivo']
y_pred = df_resultados['prediccion']

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n✅ Predicciones realizadas (solo filas con valor objetivo real):")
pd.set_option('display.max_rows', None)
print(df_resultados[['fecha', 'fecha_objetivo', 'Price', 'precio_objetivo', 'prediccion']])
pd.reset_option('display.max_rows')

print("\n📊 MÉTRICAS DEL MODELO (Cierre):")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape:.2f}%")
# Gráfico comparativo
plt.figure(figsize=(12,6))
plt.plot(df_resultados['fecha_objetivo'], df_resultados['precio_objetivo'], label='Real (t+1)', linestyle='--')
plt.plot(df_resultados['fecha_objetivo'].shift(Parametros.dias), df_resultados['prediccion'], label='Predicción (t+1)')
plt.xlabel('Fecha')
plt.ylabel('Precio WTI')
plt.title(f'Predicción del precio de cierre del WTI a {Parametros.dias} días')
plt.legend()
plt.tight_layout()
plt.show()

# Guardar predicciones a CSV
df_guardar = df_nuevo.copy()
for col in df_guardar.columns:
    if isinstance(df_guardar[col].dtype, CategoricalDtype):
        df_guardar[col] = df_guardar[col].astype(str)

output_file = Parametros.nombre+str(Parametros.dias)+"Dias.csv"
df_guardar.to_csv(output_file, index=False)
print(f"📁 Predicciones guardadas en '{output_file}'")
print("📌 Ruta completa:", os.path.abspath(output_file))

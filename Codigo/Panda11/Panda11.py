import pandas as pd
import xgboost as xgb
import Parametros
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def calcular_bollinger_bands(df, column='Price', window=20, num_std=2):
    media = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    banda_superior = media + (num_std * std)
    banda_inferior = media - (num_std * std)
    return banda_superior, banda_inferior

print("🚀 Iniciando script...")

# 1️⃣ Cargar datos
precio = pd.read_csv('../Datos/Crude Oil WTI Futures Historical Data expandido.csv', parse_dates=['fecha'])
gpr = pd.read_csv('../Datos/Gpr_por_Dia.csv', parse_dates=['fecha'])
mensual = pd.read_csv('../Datos/GPR_De_paises_por_dia.csv', parse_dates=['Date'])  
barcos = pd.read_csv('../Datos/trafico_canales1986_2025.csv', parse_dates=['Date'])
sp500=pd.read_csv('../Datos/S&P 500 Futures Historical Data expandido.csv', parse_dates=['Date'])
sp500.rename(columns={'Date': 'fecha', 'Price': 'precioSp', 'Open': 'OpenSp','High':'HighSp','Low':'LowSp','Vol.':'Vol.SP','Change':'ChangeSP'}, inplace=True)
dolar = pd.read_csv('../Datos/US Dollar Index Historical Extendido.csv', parse_dates=['fecha'])
dolar.rename(columns={'Date': 'fecha', 'Price': 'precioDolar', 'Open': 'OpenDolar','High':'HighDolar','Low':'LowDolar','Vol.':'Vol.Dolar','Change':'ChangeDolar'}, inplace=True)
dolar['fecha'] = dolar['fecha'].dt.normalize()


empresa_Petroleo = pd.read_csv('../Datos/Exxon Mobil Stock Price Expandido.csv', parse_dates=['fecha'])
empresa_Petroleo['fecha'] = empresa_Petroleo['fecha'].dt.normalize()
empresa_Petroleo.rename(columns={'Date': 'fecha', 'Price': 'PriceEP', 'Open': 'OpenEP','High':'HighEP','Low':'LowEP','Vol.':'Vol.EP','Change':'ChangeEP'}, inplace=True)

# 2️⃣ Preparar fechas
mensual.rename(columns={'Date': 'fecha'},inplace=True)
barcos.rename(columns={'Date': 'fecha', 'Panama': 'barcos_panamá', 'Suez': 'barcos_suez'}, inplace=True)
barcos['fecha'] = barcos['fecha'].dt.normalize()
sp500['fecha'] = sp500['fecha'].dt.normalize()
precio['fecha'] = precio['fecha'].dt.normalize()
gpr['fecha'] = gpr['fecha'].dt.normalize()
mensual['fecha'] = mensual['fecha'].dt.normalize()

# 🔄 Expandir mensual a diario
fechas_diarias = pd.DataFrame({'fecha': pd.date_range(start=precio['fecha'].min(), end=precio['fecha'].max(), freq='D')})
mensual_diario = fechas_diarias.merge(mensual, on='fecha', how='left').ffill().infer_objects()
barcos_diario = fechas_diarias.merge(barcos, on='fecha', how='left').ffill().infer_objects()
Sp500_diario= fechas_diarias.merge(sp500, on='fecha', how='left').ffill().infer_objects()
dolar_diario=fechas_diarias.merge(dolar, on='fecha', how='left').ffill().infer_objects()
EP_diario=fechas_diarias.merge(empresa_Petroleo, on='fecha', how='left').ffill().infer_objects()
# 3️⃣ Merge total
df = precio.merge(gpr, on='fecha').sort_values('fecha')
df = df.merge(mensual_diario, on='fecha', how='left')
df = df.merge(barcos_diario, on='fecha', how='left')
df = df.merge(Sp500_diario, on='fecha', how='left')
df= df.merge(dolar_diario, on='fecha', how='left')
df= df.merge(EP_diario, on='fecha', how='left')

print(f"✅ Filas después del merge completo: {df.shape[0]}")

# 4️⃣ Variables adicionales
df['GPRD'] = pd.to_numeric(df['GPRD'], errors='coerce')
df['GPRD_Delta'] = df['GPRD'] - df['GPRD'].shift(3)

# Media del mes pasado
precio['mes'] = precio['fecha'].dt.to_period('M')
media_mensual = precio.groupby('mes')['Price'].mean().reset_index()
media_mensual['mes'] = media_mensual['mes'].astype(str)
df['mes'] = df['fecha'].dt.to_period('M').astype(str)
df = df.merge(media_mensual.rename(columns={'Price': 'Price_Media_Mes_Pasado', 'mes': 'mes_pasado'}),
              left_on='mes', right_on='mes_pasado', how='left')
df['Price_Media_Mes_Pasado'] = df['Price_Media_Mes_Pasado'].shift(1)
df.drop(columns=['mes', 'mes_pasado'], inplace=True)

# 5️⃣ Target futuro
df['precio_objetivo'] = df['Price'].shift(-Parametros.dias)

df['Price_lag1'] = df['Price'].shift(1)
df['Price_lag2'] = df['Price'].shift(1)-df['Open'].shift(1)
df['Pendiente']=df['Price'].shift(2)-df['Price'].shift(1)
df['Alto-Bajo']=df['High'].shift(1)-df['Low'].shift(1)
df['Bajo-Alto']=df['Low'].shift(1)-df['High'].shift(1)
df['Price_lag3'] = df['Price']-df['Price'].shift(1)
df['media_col1'] = df['Price'].mean()
df['desv_col1'] = df['Price'].std()
df['skew_col1'] = df['Price'].skew()
df['kurt_col1'] = df['Price'].kurt()
df['cambio_pct'] = df['Price'].pct_change()
df['evento_extremo'] = (df['cambio_pct'].abs() > 0.05).astype(int)
df['PrecioEnteroSP']=df['precioSp']*1000+df['PriceD']
#df['OpenEnteroSP']=df['OpenSp']*1000+df['OpenD']
#df['HighEnteroSP']=df['HighSp']*1000+df['HighD']
#df['LowEnteroSP']=df['LowSp']*1000+df['LowD']
df['media_sp'] = df['PrecioEnteroSP'].mean()
df['desv_sp'] = df['PrecioEnteroSP'].std()
df['skew_sp'] = df['PrecioEnteroSP'].skew()
df['kurt_sp'] = df['PrecioEnteroSP'].kurt()
df['GPRDCambio'] = df['GPRD'].pct_change()
df['GPRD_extremo'] = (df['GPRDCambio'].abs() > 0.1).astype(int)
df['momentum_EP'] = df['PriceEP'] - df['PriceEP'].shift(4)
df['momentum_sp'] = df['PrecioEnteroSP'] - df['PrecioEnteroSP'].shift(30)
df['momentum_dolar'] = df['precioDolar'] - df['precioDolar'].shift(10) #Dinámicas muy interesantes de la tendencia del dolar 10 dias en el precio de dentro de 7 días
df['Vol_WTI'] = df['Open'].pct_change().rolling(window=8).std()
umbral_vol_wti = df['Vol_WTI'].quantile(0.97)
df['Vol_WTI_extrema'] = (df['Vol_WTI'] > umbral_vol_wti).astype(int)
df['Vol_SP'] = df['PrecioEnteroSP'].pct_change().rolling(window=5).std()
umbral_vol_sp = df['Vol_SP'].quantile(0.9)
df['Vol_SP_extrema'] = (df['Vol_SP'] > umbral_vol_sp).astype(int)

#Experimentos 


df = df.dropna(subset=['precio_objetivo'])

# 6️⃣ Conversión de tipos
cols_obj = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA30', 'GPRD_MA7']
for col in cols_obj:
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            df[col] = df[col].astype('category')

# Mensuales
columnas_mensuales = ['GPRC_EGY', 'GPRC_ISR', 'GPRC_SAU', 'GPRC_USA']
for col in columnas_mensuales:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Object restantes
cols_object = df.select_dtypes(include='object').columns
for col in cols_object:
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
        print(f"✅ '{col}' convertido a numérico.")
    except:
        df[col] = df[col].astype('category')
        print(f"ℹ️ '{col}' convertido a categoría.")

# Barcos
for col in ['barcos_suez', 'barcos_panamá']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 7️⃣ Features y target
X = df.drop(columns=['fecha','PrecioEnteroSP', 'Vol.EP','GPRDCambio','precio_objetivo','precioDolar', 
                     'HighDolar','LowDolar','OpenDolar','ChangeDolar','Price','OpenD','PriceD','precioSp','OpenSp','HighSp',
                     'HighD','LowSp','LowD',
                     'Vol.SP','PriceEP','OpenEP','HighEP','ChangeEP','LowEP','DAY','Vol_WTI','Vol_SP'])
y = df['precio_objetivo']
print(df.columns.tolist())
# 8️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)

# 9️⃣ DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# 🔟 Grid Search con GPU
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'lambda': [1, 2],
    'alpha': [0, 1],
    'objective': ['reg:squarederror'],
    'seed': [42],
    'tree_method': ['hist'],
    'device': ['cuda'],
    'eval_metric': ['rmse']
}

best_mse = float('inf')
best_params = None
best_model = None

print("🔎 Iniciando búsqueda de hiperparámetros en GPU...")

for params in ParameterGrid(param_grid):
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )
    y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
    mse = mean_squared_error(y_test, y_pred)
    print(f"Params: {params} -> MSE: {mse:.4f} (Best iteration: {model.best_iteration})")

    if mse < best_mse:
        best_mse = mse
        best_params = params
        best_model = model
        best_evals_result = evals_result

print(f"\n🏆 Mejor MSE: {best_mse:.4f} con parámetros:\n{best_params}")

# 💾 Guardar modelo
nombre_modelo = Parametros.nombre+str(Parametros.dias)+"Dias.json"
best_model.save_model(nombre_modelo)
print(f"Modelo guardado en {nombre_modelo}")

# 🎯 Predicción final
y_pred_best = best_model.predict(dtest, iteration_range=(0, best_model.best_iteration))

# 📊 Métricas
print(f"\nMétricas del mejor modelo:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_best):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_best):.4f}")

# 📈 Reales vs predichos
plt.figure(figsize=(12,6))
plt.plot(df['fecha'].iloc[-len(y_test):], y_test, label='Real')
plt.plot(df['fecha'].shift(Parametros.dias).iloc[-len(y_test):], y_pred_best, label='Predicho')
plt.xlabel('Fecha')
plt.ylabel('Precio WTI (Cierre)')
plt.title('Predicción del precio de cierre del WTI con XGBoost (Mejor Modelo GPU)')
plt.legend()
plt.tight_layout()
plt.show()

# 📉 Curva de aprendizaje
plt.figure(figsize=(10,5))
plt.plot(best_evals_result['train']['rmse'], label='Train RMSE')
plt.plot(best_evals_result['eval']['rmse'], label='Eval RMSE')
plt.xlabel('Número de iteración')
plt.ylabel('RMSE')
plt.title('Curva de aprendizaje (Mejor Modelo GPU)')
plt.legend()
plt.show()

# 🧮 Gráfico de residuales
residuals = y_test - y_pred_best

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valor real')
plt.ylabel('Residual (Error)')
plt.title('Residuals vs Valores Reales')

plt.subplot(1,2,2)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residual')
plt.title('Distribución de errores')
plt.tight_layout()
plt.show()

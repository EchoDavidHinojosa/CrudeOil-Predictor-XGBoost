import pandas as pd
import xgboost as xgb
import Parametros
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# 1️⃣ Aquí cargaremos los datos obtenidos de lo s.csv y haremos los rename correspondientes para no tener problemas
# debido a que algunos .csv son históricos de datos con nombres comunes como Price Open etc...
precio = pd.read_csv('../Datos/Crude Oil WTI Futures Historical Data expandido.csv', parse_dates=['fecha'])
barcos = pd.read_csv('../Datos/trafico_canales1986_2025.csv', parse_dates=['Date'])
sp500=pd.read_csv('../Datos/S&P 500 Futures Historical Data expandido.csv', parse_dates=['Date'])
dolar = pd.read_csv('../Datos/US Dollar Index Historical Extendido.csv', parse_dates=['fecha'])
empresa_Petroleo = pd.read_csv('../Datos/Exxon Mobil Stock Price Expandido.csv', parse_dates=['fecha'])

# Los rename ya mencionados se harán aquí
sp500.rename(columns={'Date': 'fecha', 'Price': 'precioSp', 'Open': 'OpenSp','High':'HighSp','Low':'LowSp','Vol.':'Vol.SP','Change':'ChangeSP'}, inplace=True)
dolar.rename(columns={'Date': 'fecha', 'Price': 'precioDolar', 'Open': 'OpenDolar','High':'HighDolar','Low':'LowDolar','Vol.':'Vol.Dolar','Change':'ChangeDolar'}, inplace=True)
empresa_Petroleo.rename(columns={'Date': 'fecha', 'Price': 'PriceEP', 'Open': 'OpenEP','High':'HighEP','Low':'LowEP','Vol.':'Vol.EP','Change':'ChangeEP'}, inplace=True)
barcos.rename(columns={'Date': 'fecha', 'Panama': 'barcos_panamá', 'Suez': 'barcos_suez'}, inplace=True)

# 2️⃣ Preparar fechas

empresa_Petroleo['fecha'] = empresa_Petroleo['fecha'].dt.normalize()
dolar['fecha'] = dolar['fecha'].dt.normalize()
barcos['fecha'] = barcos['fecha'].dt.normalize()
sp500['fecha'] = sp500['fecha'].dt.normalize()
precio['fecha'] = precio['fecha'].dt.normalize()

fechas_diarias = pd.DataFrame({'fecha': pd.date_range(start=precio['fecha'].min(), end=precio['fecha'].max(), freq='D')})
barcos_diario = fechas_diarias.merge(barcos, on='fecha', how='left').ffill().infer_objects()
Sp500_diario= fechas_diarias.merge(sp500, on='fecha', how='left').ffill().infer_objects()
dolar_diario=fechas_diarias.merge(dolar, on='fecha', how='left').ffill().infer_objects()
EP_diario=fechas_diarias.merge(empresa_Petroleo, on='fecha', how='left').ffill().infer_objects()


# 3️⃣ Aquí ordeamos por fechas e indicamos cuantas filas quedan despues del merge
# hay veces que por errores de los csv o de los merge hay menos de lo normal por lo que sirve para detectar errores
df = precio.sort_values('fecha')
df = df.merge(barcos_diario, on='fecha', how='left')
df = df.merge(Sp500_diario, on='fecha', how='left')
df= df.merge(dolar_diario, on='fecha', how='left')
df= df.merge(EP_diario, on='fecha', how='left')

print(f"✅ Filas después del merge completo: {df.shape[0]}")

#4️⃣ Variables adicionales extraidas de las normales de los .csv


#Esta parte no sería necesaria con los .csv que estan de base pero los he dejado por si prefieres  modificarlos
precio['mes'] = precio['fecha'].dt.to_period('M')
media_mensual = precio.groupby('mes')['Price'].mean().reset_index()
media_mensual['mes'] = media_mensual['mes'].astype(str)
df['mes'] = df['fecha'].dt.to_period('M').astype(str)
df = df.merge(media_mensual.rename(columns={'Price': 'Price_Media_Mes_Pasado', 'mes': 'mes_pasado'}),
              left_on='mes', right_on='mes_pasado', how='left')
df['Price_Media_Mes_Pasado'] = df['Price_Media_Mes_Pasado'].shift(1)
df.drop(columns=['mes', 'mes_pasado'], inplace=True)

#Haré referencia a "ayer"(shift(1)) o hace x días(shift(x))tomando como referencia el día en el que se encuentra el modelo

# Aquí tendremos la variable a predecir(si quieres predecir en tiempo real deberás de rellenar inventandote algunos datos apra que exista un precio_objetivo 
# al que hacer referencia para predecir)
df['precio_objetivo'] = df['Price'].shift(-Parametros.dias)
#Price_lag1 da referencia al modelo del precio que había "ayer"
df['Price_lag1'] = df['Price'].shift(1)
#Price_lag2 da referencia al modelo de la diferencia entre el precio de apertura y el precio de cierre de "ayer" viendo el cambio que hay en un solo día
df['Price_lag2'] = df['Price'].shift(1)-df['Open'].shift(1)
#La diferencia del precio de hace dos días y de hace un día marca la tendencia del mercado
df['Pendiente']=df['Price'].shift(2)-df['Price'].shift(1)
#La diferencia del precio más alto y el más bajo ayer marca lo variable que ha sido el precio
df['Alto-Bajo']=df['High'].shift(1)-df['Low'].shift(1)
#Aqui repetimos la variable pero al revés para que el modelo tenga la otra medida en negativo y pueda darle un peso(da mejores resultados la rebundancia)
df['Bajo-Alto']=df['Low'].shift(1)-df['High'].shift(1)
#La diferencia del precio de cierre de "hoy" y "ayer" da un dato sobre la tendencia
df['Price_lag3'] = df['Price']-df['Price'].shift(1)
#Esto agrega el mismo valor promedio a todas las filas, útil como referencia estática para comparar cada precio individual contra la media global.
df['media_col1'] = df['Price'].mean()
#También es un valor fijo replicado en todas las filas. Sirve para entender cuán volátil ha sido históricamente el precio.
df['desv_col1'] = df['Price'].std()
#Un valor positivo indica una cola más larga a la derecha (precios muy altos ocasionales), y negativo indica lo contrario. Ayuda a detectar sesgos en la distribución.
df['skew_col1'] = df['Price'].skew()
#Una curtosis alta puede significar que hay muchos valores extremos (picos o caídas repentinas), importante en análisis financiero.
df['kurt_col1'] = df['Price'].kurt()
#Sirve para ver la variación relativa entre cada día y el anterior. Es una base común para indicadores como retornos diarios o señales de trading.
df['cambio_pct'] = df['Price'].pct_change()
#sta línea detecta eventos extremos en los cambios porcentuales del precio y crea una nueva columna llamada 'evento_extremo' con un 1 si hay un evento extremo, o 0 si no.
df['evento_extremo'] = (df['cambio_pct'].abs() > 0.05).astype(int)
#Esto es la creación de un dato debido a un problema de formato de los .csv
df['PrecioEnteroSP']=df['precioSp']*1000+df['PriceD']
#df['OpenEnteroSP']=df['OpenSp']*1000+df['OpenD']
#df['HighEnteroSP']=df['HighSp']*1000+df['HighD']
#df['LowEnteroSP']=df['LowSp']*1000+df['LowD']

#Lo mismo  que con 'Price' haciendo referencia el wti crude oil pero con el S&P500  
df['media_sp'] = df['PrecioEnteroSP'].mean()
df['desv_sp'] = df['PrecioEnteroSP'].std()
df['skew_sp'] = df['PrecioEnteroSP'].skew()
df['kurt_sp'] = df['PrecioEnteroSP'].kurt()
#Calculamos la tendencia del precio de Exxon Mobil Stock que tiene correlación en el wti crude oil
df['momentum_EP'] = df['PriceEP'] - df['PriceEP'].shift(4)
#Calculamos la tendencia del SO&P500 últimos 30 día,ya que esta compuesto por empresas que dependen de él y en caso de ser muy bueno puede reducir la especulación
# sobre el wti crude oil afectando e este
df['momentum_sp'] = df['PrecioEnteroSP'] - df['PrecioEnteroSP'].shift(30)
#Calculamos la tendencia del dolar de los últimos 10 día spuesto que un cambio abrupto en el dolar puede repercutir en el WTI oil crude
df['momentum_dolar'] = df['precioDolar'] - df['precioDolar'].shift(10) #Dinámicas muy interesantes de la tendencia del dolar 10 dias en el precio de dentro de 7 días

#Aqui calculamos la volatilidad en caso de haber un pico 
df['Vol_WTI'] = df['Open'].pct_change().rolling(window=8).std()
umbral_vol_wti = df['Vol_WTI'].quantile(0.97)
df['Vol_WTI_extrema'] = (df['Vol_WTI'] > umbral_vol_wti).astype(int)
#Unos dátos estáticos del dolar paara tener de referencia
df['media_Dolar'] = df['precioDolar'].mean()
df['desv_Dolar'] = df['precioDolar'].std()
df['skew_Dolar'] = df['precioDolar'].skew()






df = df.dropna(subset=['precio_objetivo'])




# 5️⃣Object restantes
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
X = df.drop(columns=['fecha','PrecioEnteroSP', 'Vol.EP','precio_objetivo','precioDolar', 
                     'HighDolar','LowDolar','OpenDolar','ChangeDolar','Open','OpenD','PriceD','precioSp','OpenSp','HighSp',
                     'HighD','LowSp','LowD',
                     'Vol.SP','PriceEP','OpenEP','HighEP','ChangeEP','LowEP','Vol_WTI'])
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

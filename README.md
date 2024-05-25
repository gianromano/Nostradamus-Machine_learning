NOSTRADAMUS v1.3
Es una herramienta creada por y para analistas de datos con la finalidad de ofrecer una interface para interactuar con un data set y entrenar un modelo de aprendisaje mecánico (machine learning).  

 - Entre sus capcacidades esta la de poder interactuar con cualquier tipo de dataset que haya sido previamente limpiado y balanceado.
 - Realiza carga de datos desde ficheros csv.
 - Permite llevar a cabo el proceso distinguiendo entre datos discretos y datos continuos.
 - Permite desechar columnas que no se utilizarán en el entrenamiento.
 - Permite elegir la columna target.
 - Muestra esquema descriptivo con información de la media, desviación estandar, percentiles, mínimos y máximos de las diferentes columnas del dataset.
 - Realiza gráfica de correlaciones para facilitar la selección de caracterísiticas (features) del modelo.
 - Permite realizar porcesos de transformación de datos: Discretización y One-Hand Encoding.
- Ofrece los siguientes modelos:
  Para valores discretos: KNN Classifier, Regresión Logística, Bagging (clasificadores), Random Forest (clasificación), Adaptive Boost, Gradient Boost (clasificación).
  Para valores continuos: KNN Regresor, Regresión Lineal, Árbol de Decisión (regresión), Bagging (regresores), Random Forest (regresión), Gradient Boost (regresión).
- Normalizadores como: MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler. Adecuados para trabajar con valores continuos.
- CrossValidation:
  Para valores discretos (clasificación): Stratified K-Fold, Stratified Shuffle Split
  Para valores continuos (regresión): K-Fold, Shuffle Split
- Hyperparemetrización de modelos con: 
  Grid Search: Apto para valores continuos y valores discretos.
  Random Search: Apto para valores continuos y valores discretos.

En el nostradamus.py que se encuentra en este repositorio podemos encontrar las funciones implementadas en este proyecto para llevar a cabo la ejecución en Streamlit.

La autoría de este proyecto pertenece a Gian Romano.

Proximas versiones incluirán:
Modelos :Árbol de Decisión (clasificación), Support Vector Machine, Neural Networks (clasificación), Neural Networks (regresión).
CrossValidation: Group K-Fold, Group Shuffle Split

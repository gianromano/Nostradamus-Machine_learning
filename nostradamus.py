import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    mean_absolute_error, mean_squared_error, make_scorer
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    KFold, GroupKFold, ShuffleSplit, StratifiedKFold, 
    GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
)

# Función para cargar datos
def load_data():
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv", key="file_uploader")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', quotechar='"')
            st.write("Datos cargados:")
            st.write(df)
            return df
        except ValueError as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

# Función para limpiar datos
def clean_data(df):
    st.write("Columnas disponibles:")
    selected_columns = st.multiselect("Selecciona las columnas a eliminar:", [""] +  df.columns.tolist())
    df = df.drop(columns=selected_columns, axis=1)
    st.write("Descripción de los datos después de eliminar columnas:")
    st.write(df.describe())
    return df

def select_target_column(df):
    target_column = st.selectbox("Elige la columna objetivo", [""] + list(df.columns))
    return target_column

def ask_discretization_or_one_hot_encoding():
    option = st.selectbox("Elige método de transformación de datos", ["", "Discretizar", "One-Hot Encoding"])
    return option

def discretizing_data(df):
    df_discretized = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            df[col] = pd.cut(df[col], bins=5, labels=["low", "mid-low", "medium", "mid-high", "high"])
    st.write(df_discretized)   
    return df_discretized

def one_hot_encoder(df):
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    st.write(df_encoded)
    return df_encoded

def target_features(df, target_column):
    features = df.drop(columns=[target_column])
    target = df[target_column] 
    return features, target 

def feature_selection(df):
    corr = np.abs(df.corr())

    # Set up mask for triangle representation
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=corr, cmap=cmap)

    st.pyplot(f)
    
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test    

def ask_data_type():
    data_type = st.radio("¿Elige el tipo de datos?", ("Discretos", "Continuos"), index=None)
    return data_type

# Función para normalización de valores
def normalize_data(X_train, X_test):
    # Opciones de normalización
    normalizers = {
        "Min-Max Scaler": MinMaxScaler(),
        "Standard Scaler": StandardScaler(),
        "Robust Scaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "Max-Abs Scaler": MaxAbsScaler()
    }
    
    # Selección del método de normalización con opción en blanco
    normalizer_option = st.selectbox("Elige un método de normalización", [""] + list(normalizers.keys()), key="normalizer_selector")
    
    if normalizer_option == "":
        st.stop()
    
    # Obtener el normalizador seleccionado
    normalizer = normalizers[normalizer_option]
    normalizer.fit(X_train)

    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    # Convertir los resultados a DataFrame para mantener las columnas originales
    X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns)
    X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns)

    st.write("Valores normalizados usando:", normalizer_option)
    st.write(X_train_norm.head())

    return X_train_norm, X_test_norm

def select_cv_method_discrete():
    cv_method = st.selectbox("Selecciona un método de validación cruzada para datos discretos", ["Stratified K-Fold", "Stratified Shuffle Split"])
    return cv_method

def select_cv_method_continuous():
    cv_method = st.selectbox("Selecciona un método de validación cruzada para datos continuos", ["K-Fold", "Shuffle Split"])
    return cv_method


def cross_validation_discrete(model, X, y, cv_method):
    if cv_method == "Stratified K-Fold":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_method == "Stratified Shuffle Split":
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    else:
        raise ValueError("Método de validación cruzada no válido.")
    scoring = make_scorer(accuracy_score)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores, cv

def cross_validation_continuous(model, X, y, cv_method):
    groups = None  # Inicializa los grupos como None

    if cv_method == "K-Fold":
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_method == "Shuffle Split":
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # elif cv_method == "Group K-Fold":
    #     groups = np.random.randint(1, 10, size=len(y))  # Ejemplo de grupos aleatorios
    #     cv = GroupKFold(n_splits=5)
    # elif cv_method == "Group Shuffle Split":
    #     groups = np.random.randint(1, 10, size=len(y))  # Ejemplo de grupos aleatorios
    #     cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    else:
        raise ValueError("Método de validación cruzada no válido para datos continuos")

    scoring = "neg_mean_squared_error"

    if groups is not None:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, groups=groups)
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    if scoring == "neg_mean_squared_error":
        scores = -scores  # Negativiza el MSE para que sea positivo

    return scores, cv


def choose_hyperparameters_discrete(model):
    # Opciones para la búsqueda de hiperparámetros
    hyperparam_options = {
        "Grid Search": "grid",
        "Random Search": "random"
    }

    hyperparam_option = st.selectbox("Elige una opción de búsqueda de hiperparámetros", [""] + list(hyperparam_options.keys()))

    # Definir los parámetros dependiendo del modelo seleccionado
    if isinstance(model, KNeighborsClassifier):
        if hyperparam_option == "Grid Search":
            n_neighbors = st.slider("Número de vecinos", min_value=1, max_value=50, value=5)
            weights = st.selectbox("Pesos", ["uniform", "distance"])
            algorithm = st.selectbox("Algoritmo", ["auto", "ball_tree", "kd_tree", "brute"])

            params = {
                'n_neighbors': [n_neighbors],
                'weights': [weights],
                'algorithm': [algorithm]
            }
        else:  # Si es Random Search
            params = {
                'n_neighbors': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

    elif isinstance(model, LogisticRegression):
        if hyperparam_option == "Grid Search":
            penalty = st.selectbox("Penalización", ["l1", "l2", "elasticnet", "none"])
            C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0)
            solver = st.selectbox("Algoritmo de optimización", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])

            params = {
                'penalty': [penalty],
                'C': [C],
                'solver': [solver]
            }
        else:  # Si es Random Search
            params = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

    elif isinstance(model, BaggingClassifier):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, step=1, value=10)
            max_samples = st.number_input("Número máximo de muestras", min_value=0.1, max_value=1.0, step=0.1, value=1.0)
            max_features = st.number_input("Número máximo de características", min_value=0.1, max_value=1.0, step=0.1, value=1.0)

            params = {
                'n_estimators': [n_estimators],
                'max_samples': [max_samples],
                'max_features': [max_features]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                'max_samples': [0.1, 0.5, 0.7, 0.9, 1.0],
                'max_features': [0.1, 0.5, 0.7, 0.9, 1.0]
            }

    elif isinstance(model,RandomForestClassifier):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, step=1, value=100)
            criterion = st.selectbox("Criterio", ["gini", "entropy"])
            max_depth = st.number_input("Profundidad máxima", min_value=1, step=1, value=None)

            params = {
                'n_estimators': [n_estimators],
                'criterion': [criterion],
                'max_depth': [max_depth]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 40, 50]
            }

    elif isinstance(model,AdaBoostClassifier):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, step=1, value=50)
            learning_rate = st.number_input("Tasa de aprendizaje", min_value=0.01, max_value=1.0, step=0.01, value=1.0)
            algorithm = st.selectbox("Algoritmo", ["SAMME", "SAMME.R"])

            params = {
                'n_estimators': [n_estimators],
                'learning_rate': [learning_rate],
                'algorithm': [algorithm]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }

    elif isinstance(model,GradientBoostingClassifier):
        if hyperparam_option == "Grid Search":    
            learning_rate = st.number_input("Tasa de aprendizaje", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
            n_estimators = st.number_input("Número de estimadores", min_value=1, step=1, value=100)
            max_depth = st.number_input("Profundidad máxima", min_value=1, step=1, value=3)

            params = {
                'learning_rate': [learning_rate],
                'n_estimators': [n_estimators],
                'max_depth': [max_depth]
            }
        else:  # Si es Random Search
            params = {
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'n_estimators': [10, 50, 100, 200, 300, 400, 500],
                'max_depth': [3, 5, 10, 20, None]
            }


    else:
        st.error("Modelo no válido para la búsqueda de hiperparámetros")

    return params, hyperparam_option


def choose_hyperparameters_continuous(model):
    # Opciones para la búsqueda de hiperparámetros
    hyperparam_options = {
        "Grid Search": "grid",
        "Random Search": "random"
    }
    
    hyperparam_option = st.selectbox("Elige una opción de búsqueda de hiperparámetros", [""] + list(hyperparam_options.keys()))

    # Definir los parámetros dependiendo del modelo y la opción seleccionada
    if isinstance(model, LinearRegression):
        if hyperparam_option == "Grid Search":
            fit_intercept = st.selectbox("Fit Intercept", [True, False])
            copy_X = st.selectbox("Copy X", [True, False])
            positive = st.selectbox("Positive", [True, False])

            params = {
                'fit_intercept': [fit_intercept],
                'copy_X': [copy_X],
                'positive': [positive]
            }
        else:  # Si es Random Search
            params = {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'positive': [True, False]
            }

    elif isinstance(model, KNeighborsRegressor):
        if hyperparam_option == "Grid Search":
            n_neighbors = st.number_input("Número de vecinos", min_value=1, max_value=30, step=1)
            algorithm = st.selectbox("Algoritmo", ['auto', 'ball_tree', 'kd_tree', 'brute'])
            weights = st.selectbox("Pesos", ['uniform', 'distance'])

            params = {
                'n_neighbors': [n_neighbors],
                'algorithm': [algorithm],
                'weights': [weights]
            }
        else:  # Si es Random Search
            params = {
                'n_neighbors': [int(x) for x in range(1, 31)],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'weights': ['uniform', 'distance']
            }

    elif isinstance(model, DecisionTreeRegressor):
        if hyperparam_option == "Grid Search":
            criterion = st.selectbox("Criterio", ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'])
            max_depth = st.number_input("Profundidad máxima", min_value=1, max_value=50, step=1)
            min_samples_split = st.number_input("Mínimo de muestras para dividir", min_value=2, max_value=10, step=1)
            min_samples_leaf = st.number_input("Mínimo de muestras en hojas", min_value=1, max_value=10, step=1)

            params = {
                'criterion': [criterion],
                'max_depth': [max_depth],
                'min_samples_split': [min_samples_split],
                'min_samples_leaf': [min_samples_leaf]
                }
        else:  # Si es Random Search
            params = {
                'criterion': ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'],
                'max_depth': [int(x) for x in range(1, 51)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif isinstance(model, BaggingRegressor):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, max_value=100, step=1)
            max_samples = st.number_input("Número máximo de muestras", min_value=0.1, max_value=1.0, step=0.1, value=1.0)
            max_features = st.number_input("Número máximo de características", min_value=0.1, max_value=1.0, step=0.1, value=1.0)

            params = {
                'n_estimators': [n_estimators],
                'max_samples': [max_samples],
                'max_features': [max_features]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [int(x) for x in range(1, 101)],
                'max_samples': [float(x)/10 for x in range(1, 11)],
                'max_features': [float(x)/10 for x in range(1, 11)]
            }
    elif isinstance(model, RandomForestRegressor):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, max_value=200, step=1)
            max_depth = st.number_input("Profundidad máxima", min_value=1, max_value=50, step=1)
            min_samples_split = st.selectbox("Mínimo de muestras para dividir", [2, 5, 10])
            min_samples_leaf = st.selectbox("Mínimo de muestras en hojas", [1, 2, 4])

            params = {
                'n_estimators': [n_estimators],
                'max_depth': [max_depth],
                'min_samples_split': [min_samples_split],
                'min_samples_leaf': [min_samples_leaf]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [int(x) for x in range(1, 201)],
                'max_depth': [int(x) for x in range(1, 51)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif isinstance(model, AdaBoostRegressor):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, max_value=200, step=1)
            learning_rate = st.number_input("Tasa de aprendizaje", min_value=0.01, max_value=2.0, step=0.01)

            params = {
                'n_estimators': [n_estimators],
                'learning_rate': [learning_rate]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [int(x) for x in range(1, 201)],
                'learning_rate': [float(x)/10 for x in range(1, 21)]
            }
    elif isinstance(model, GradientBoostingRegressor):
        if hyperparam_option == "Grid Search":
            n_estimators = st.number_input("Número de estimadores", min_value=1, max_value=200, step=1)
            learning_rate = st.number_input("Tasa de aprendizaje", min_value=0.01, max_value=2.0, step=0.01)
            max_depth = st.number_input("Profundidad máxima", min_value=1, max_value=50, step=1)

            params = {
                'n_estimators': [n_estimators],
                'learning_rate': [learning_rate],
                'max_depth': [max_depth]
            }
        else:  # Si es Random Search
            params = {
                'n_estimators': [int(x) for x in range(1, 201)],
            'learning_rate': [float(x)/10 for x in range(1, 21)],
            'max_depth': [int(x) for x in range(1, 51)]
        }

    else:
        st.error("Modelo no soportado para hiperparametrización")
        return None, None

    return params, hyperparam_option

def hyperparameter_search_discrete(model, hyperparam_option, params, cv, X_train, y_train):
    # Obtener el objeto de búsqueda de hiperparámetros seleccionado
    if hyperparam_option == "Grid Search":
        hyperparam_search = GridSearchCV
    elif hyperparam_option == "Random Search":
        hyperparam_search = RandomizedSearchCV
    else:
        raise ValueError("Opción de búsqueda de hiperparámetros no válida")

    # Realizar la búsqueda de hiperparámetros
    search = hyperparam_search(model, params, cv=cv)
    search.fit(X_train, y_train)

    # Obtener y devolver el mejor modelo
    best_model = search.best_estimator_
    return best_model

def hyperparameter_search_continuous(model, hyperparam_option, params, cv, X_train_norm, y_train):
    # Obtener el objeto de búsqueda de hiperparámetros seleccionado
    if hyperparam_option == "Grid Search":
        hyperparam_search = GridSearchCV
    elif hyperparam_option == "Random Search":
        hyperparam_search = RandomizedSearchCV
    else:
        raise ValueError("Opción de búsqueda de hiperparámetros no válida")

    # Realizar la búsqueda de hiperparámetros
    search = hyperparam_search(model, params, cv=cv)
    search.fit(X_train_norm, y_train)

    # Obtener y devolver el mejor modelo
    best_model = search.best_estimator_
    return best_model


def choose_model_discrete_discretized(X_train, X_test, y_train, y_test, data_type):
    model_option = st.selectbox("Elige un modelo", ["","Bagging Classifier", "Random Forest Classifier", "AdaBoost Classifier", "Gradient Boost Classifier"], key="model_selector")
    model = None  # Inicializar el modelo
    
    if model_option == "Bagging Classifier":
        model = BaggingClassifier(n_estimators=100, max_samples=0.5)
        BaggingClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=20)
        RandomForestClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "AdaBoost Classifier":
        model = AdaBoostClassifier(n_estimators=100)
        AdaBoostClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Gradient Boost Classifier":
        model = GradientBoostingClassifier(max_depth=20, n_estimators=100)
        GradientBoostingClassifier_model(model, X_train, X_test, y_train, y_test)
        
    return model  # Devolver el modelo

def choose_model_discrete_one_hot_encoded(X_train, X_test, y_train, y_test, data_type):
    model_option = st.selectbox("Elige un modelo", ["","KNN Clasifier", "Logistic Regression", "Bagging Classifier", "Random Forest Classifier", "AdaBoost Classifier", "Gradient Boost Classifier"], key="model_selector")
    model = None  # Inicializar el modelo
    
    if model_option == "KNN Clasifier":
        model = KNeighborsClassifier(n_neighbors=3)
        KNeighborsClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
        LogisticRegression_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Bagging Classifier":
        model = BaggingClassifier(n_estimators=100, max_samples=0.5)
        BaggingClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, max_depth=20)
        RandomForestClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "AdaBoost Classifier":
        model = AdaBoostClassifier(n_estimators=100)
        AdaBoostClassifier_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Gradient Boost Classifier":
        model = GradientBoostingClassifier(max_depth=20, n_estimators=100)
        GradientBoostingClassifier_model(model, X_train, X_test, y_train, y_test)
        
    return model  # Devolver el modelo

def choose_model_continuos(X_train_norm, X_test_norm, y_train, y_test, data_type):
    model_option = st.selectbox("Elige un modelo", ["","KNN Regressor", "Linear Regressor", "Decision Tree Regressor", "Bagging Regressor", "Random Forest Regressor", "AdaBoost Regressor", "Gradient Boost Regressor"], key="model_selector")
    model = None  # Inicializar el modelo
    
    if model_option == "KNN Regressor":
        model = KNeighborsRegressor(n_neighbors=10)
        KNeighborsRegressor_model(model, X_train, X_test, y_train, y_test)
    elif model_option == "Linear Regressor":
        model = LinearRegression()
        LinearRegression_model(model, X_train_norm, X_test_norm, y_train, y_test)
    elif model_option == "Decision Tree Regressor":
        model = DecisionTreeRegressor(max_depth=10)
        DecisionTreeRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test)
    elif model_option == "Bagging Regressor":
        model = BaggingRegressor(n_estimators=100, max_samples=0.5)
        BaggingRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test)
    elif model_option == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, max_depth=20)
        RandomForestRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test)
    elif model_option == "AdaBoost Regressor":
        model = AdaBoostRegressor(n_estimators=100)
        AdaBoostRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test)        
    elif model_option == "Gradient Boost Regressor":
        model = GradientBoostingRegressor(max_depth=20, n_estimators=100)
        GradientBoostingRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test)
        
    return model  # Devolver el modelo

# Function to train and evaluate the KNeighborsClassifier model with discretized data
def KNeighborsClassifier_model(model, X_train, X_test, y_train, y_test):
    # Encode the categorical features and target
    le = LabelEncoder()
    
    X_train_encoded = X_train.apply(le.fit_transform)
    X_test_encoded = X_test.apply(le.transform)
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    model.fit(X_train_encoded, y_train_encoded)
    y_pred = model.predict(X_test_encoded)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test_encoded, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test_encoded, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test_encoded, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Función para entrenar y evaluar el KNeighborsRegressor_model con métricas y gráficos
def KNeighborsRegressor_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    st.write("MAE:", mean_absolute_error(pred, y_test))
    st.write("RMSE:", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score:", model.score(X_test, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)

# Función para entrenar y evaluar el LogisticRegression model with discretized data
def LogisticRegression_model(model, X_train, X_test, y_train, y_test):
    # Encode the categorical features and target
    le = LabelEncoder()
    
    X_train_encoded = X_train.apply(le.fit_transform)
    X_test_encoded = X_test.apply(le.transform)
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    model.fit(X_train_encoded, y_train_encoded)
    y_pred = model.predict(X_test_encoded)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test_encoded, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test_encoded, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test_encoded, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Función para entrenar y evaluar el LinearRegression_model NORMALIZE
def LinearRegression_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)

    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)



# Función para entrenar y evaluar el DecisionTreeRegressor_model NORMALIZE
def DecisionTreeRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)

    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)

# Function to train and evaluate the BaggingClassifier model with discretized data
def BaggingClassifier_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Function to train and evaluate the BaggingRegressor model with normalized data
def BaggingRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)
    
    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)

# Function to train and evaluate the RandomForestClassifier model with discretized data
def RandomForestClassifier_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Function to train and evaluate the RandomForestRegressor model with normalized data
def RandomForestRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)

    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)

# Function to train and evaluate the AdaBoostClassifier model with discretized data
def AdaBoostClassifier_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Función para entrenar y evaluar el AdaBoostRegressor_model NORMALIZE
def AdaBoostRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)
    
    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)

# Function to train and evaluate the GradientBoostingClassifier model with discretized data
def GradientBoostingClassifier_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Matriz de Confusión: ")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    st.write("Reporte de Clasificación: ")
    cr = classification_report(y_test, y_pred)
    st.text(cr)
    
    st.write("Precisión del modelo: ", accuracy_score(y_test, y_pred))
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Función para entrenar y evaluar el GradientBoostingRegressor_model NORMALIZE
def GradientBoostingRegressor_model(model, X_train_norm, X_test_norm, y_train, y_test):
    model.fit(X_train_norm, y_train)
    pred = model.predict(X_test_norm)

    st.write("MAE", mean_absolute_error(pred, y_test))
    st.write("RMSE", mean_squared_error(pred, y_test, squared=False))
    st.write("R2 score", model.score(X_test_norm, y_test))

    # Crear gráfico de dispersión
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    st.pyplot(fig)


# Inicio de la aplicacion
st.title("NOSTRADAMUS")

df = load_data()


if df is not None:

    data_type = ask_data_type()

    df = clean_data(df)    
    feature_selection(df)

    target_column = select_target_column(df)

    if target_column != "":

        if data_type == "Discretos":
            option = ask_discretization_or_one_hot_encoding()
            if option == "Discretizar":
                df_discretized = discretizing_data(df)
                features, target = target_features(df_discretized, target_column)
                X_train, X_test, y_train, y_test = split_data(features, target)
                model = choose_model_discrete_discretized(X_train, X_test, y_train, y_test, data_type)

                if model is not None:
                    cv_method = select_cv_method_discrete()
                    scores, cv = cross_validation_discrete(model, X_train, y_train, cv_method)
                    params, hyperparam_option = choose_hyperparameters_discrete(model)

                    if hyperparam_option != "":   
                        best_model = hyperparameter_search_discrete(model, hyperparam_option, params, cv, X_train, y_train)
                        st.write(best_model)

            elif option == "One-Hot Encoding":
                df_encoded = one_hot_encoder(df)
                features, target = target_features(df_encoded, target_column)
                X_train, X_test, y_train, y_test = split_data(features, target)
                model = choose_model_discrete_one_hot_encoded(X_train, X_test, y_train, y_test, data_type)

                if model is not None:
                    cv_method = select_cv_method_discrete()
                    scores, cv = cross_validation_discrete(model, X_train, y_train, cv_method)
                    params, hyperparam_option = choose_hyperparameters_discrete(model)

                    if hyperparam_option != "":   
                        best_model = hyperparameter_search_discrete(model, hyperparam_option, params, cv, X_train, y_train)
                        st.write(best_model)
        
        elif data_type == "Continuos":
            features, target = target_features(df, target_column)
            X_train, X_test, y_train, y_test = split_data(features, target)
            X_train_norm, X_test_norm = normalize_data(X_train, X_test)
            
            if not X_train_norm.empty and not X_test_norm.empty:
                model = choose_model_continuos(X_train_norm, X_test_norm, y_train, y_test, data_type)
                if model is not None:
                    cv_method = select_cv_method_continuous()
                    scores, cv = cross_validation_continuous(model, X_train_norm, y_train, cv_method)
                    params, hyperparam_option = choose_hyperparameters_continuous(model)

                    if hyperparam_option != "":   
                        best_model = hyperparameter_search_continuous(model, hyperparam_option, params, cv, X_train_norm, y_train)
                        st.write(best_model)

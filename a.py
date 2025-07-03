from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Obtener el dataset
adult = fetch_ucirepo(id=2)

# Datos 
X = adult.data.features
y = adult.data.targets

# Preprocesamiento
# Imputar las columnas numéricas con la media
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Imputar las columnas categóricas con la moda 
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Convertir las variables categóricas a dummies 
X = pd.get_dummies(X, drop_first=True)

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir los modelos
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Definir las grillas de hiperparámetros para GridSearchCV 
param_grids = {
    'KNN': {'n_neighbors': [3, 5]},  
    'SVM': {'C': [0.1, 1], 'kernel': ['linear']},  
    'Logistic Regression': {'C': [0.1, 1]},  
    'Decision Tree': {'max_depth': [5, 10]},  
    'Random Forest': {'n_estimators': [50], 'max_depth': [5, 10]}  
}

# Búsqueda en cuadrícula y evaluación de cada modelo 
best_models = {}
for name, model in models.items():
    print(f"Entrenando {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1)  
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejores parámetros para {name}: {grid_search.best_params_}")
    print(f"Precisión para {name}: {accuracy}\n")

best_model_name = max(best_models, key=lambda x: accuracy_score(y_test, best_models[x].predict(X_test)))
best_model = best_models[best_model_name]
print(f"El mejor modelo es: {best_model_name} con precisión: {accuracy_score(y_test, best_model.predict(X_test))}")

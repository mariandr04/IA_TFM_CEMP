# Instalar librerías necesarias
!pip install imbalanced-learn

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from google.colab import drive
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Montar Google Drive
drive.mount('/content/drive')

# Leer el archivo CSV
file_path = '/content/drive/My Drive/Colab Notebooks/heart.csv'
df = pd.read_csv(file_path)

# Mostrar primeras filas
print("Primeras filas del dataset:")
display(df.head())

# Información general
print("\nInformación del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
display(df.describe())




# -------------------------------------------
# 1. Binarizar la columna 'num' sin alterar el original
# -------------------------------------------
df_bin = df.copy()
# Histogramas + KDE de variables numéricas
numeric_cols = df.select_dtypes(include='number').columns.drop('num')
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()
# ——————————————
df_bin['num_binarizada'] = df_bin['num'].apply(lambda x: 1 if x > 0 else 0)

# Eliminar la columna original 'num'
df_bin.drop(columns=['num'], inplace=True)

# Calcular proporciones
tasa = df_bin['num_binarizada'].value_counts(normalize=True).sort_index()

labels = ['Sano (0)', 'Enfermo (1)']

# Dibujar barplot
plt.figure(figsize=(6,4))
ax = sns.barplot(x=tasa.index, y=tasa.values, palette='Blues')
plt.xticks([0,1], labels)
plt.ylabel('Proporción')
plt.ylim(0,1)
plt.title('Proporción de variable objetivo(num) con porcentaje')

# Anotar porcentaje
for p, pct in zip(ax.patches, tasa.values):
    ax.annotate(f'{pct*100:.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom',
                fontsize=12)

plt.tight_layout()
plt.show()

# -------------------------------------------
# 2. Análisis de variables
# -------------------------------------------
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for var in categorical_vars:
    print(f"\n--- Análisis de la variable '{var}' ---")

    # Mostrar valores únicos y conteos
    print(f"Valores únicos: {df_bin[var].unique()}")
    print(df_bin[var].value_counts())

    # Gráfico de barras agrupado por clase de enfermedad
    plt.figure(figsize=(6, 4))
    sns.countplot(x=var, hue='num_binarizada', data=df_bin, palette='Set2')
    plt.title(f'Relación entre {var} y num_binarizada')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Tasa de enfermedad por categoría
    tasa = df_bin.groupby(var)['num_binarizada'].mean().round(3)
    print(f"Tasa de enfermedad cardíaca (proporción de casos positivos) por categoría en '{var}':\n{tasa}")


# —————— Visualización específica para 'cp' y 'exang' ——————

# 1) Countplot agrupado de 'cp'
plt.figure(figsize=(6,4))
sns.countplot(
    data=df_bin,
    x='cp',
    hue='num_binarizada',
    palette='Set2'
)
plt.title('Distribución de dolor torácico (cp) por clase')
plt.xlabel('Tipo de dolor torácico (cp)')
plt.ylabel('Frecuencia')
plt.legend(title='Enfermedad', labels=['Sano (0)', 'Enfermo (1)'])
plt.tight_layout()
plt.show()

# 2) Barplot de proporción enfermos por categoría 'cp'
tasa_cp = df_bin.groupby('cp')['num_binarizada'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(
    data=tasa_cp,
    x='cp',
    y='num_binarizada',
    palette='Blues'
)
plt.title('Tasa de enfermedad cardíaca por tipo de dolor torácico')
plt.xlabel('Tipo de dolor torácico (cp)')
plt.ylabel('Proporción de enfermos')
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# 3) Countplot agrupado de 'exang'
plt.figure(figsize=(6,4))
sns.countplot(
    data=df_bin,
    x='exang',
    hue='num_binarizada',
    palette='Set2'
)
plt.title('Angina inducida por ejercicio (exang) por clase')
plt.xlabel('exang (0 = No, 1 = Sí)')
plt.ylabel('Frecuencia')
plt.legend(title='Enfermedad', labels=['Sano (0)', 'Enfermo (1)'])
plt.tight_layout()
plt.show()

# 4) Barplot de proporción enfermos por categoría 'exang'
tasa_exang = df_bin.groupby('exang')['num_binarizada'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(
    data=tasa_exang,
    x='exang',
    y='num_binarizada',
    palette='Blues'
)
plt.title('Tasa de enfermedad cardíaca según angina inducida por ejercicio')
plt.xlabel('exang (0 = No, 1 = Sí)')
plt.ylabel('Proporción de enfermos')
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# Definir el tamaño de la grilla
n_cols = 4
n_rows = (len(categorical_vars) + n_cols - 1) // n_cols  # número de filas necesarias

plt.figure(figsize=(5 * n_cols, 4 * n_rows))  # tamaño total del lienzo

for i, var in enumerate(categorical_vars, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=var, hue='num_binarizada', data=df_bin, palette='Set2')
    plt.title(f'{var} vs num_binarizada')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

plt.suptitle('Distribución de variables categóricas por clase', fontsize=16, y=1.02)
plt.show()


# -------------------------------------------
# 3. Mapa de calor con la matriz de correlación
# -------------------------------------------
corr_matrix = df_bin.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Mapa de calor de correlación entre variables (con num_binarizada)")
plt.tight_layout()
plt.show()


# -------------------------------------------
# 4.Gráfico de Barras de Coeficientes de Correlación
# -------------------------------------------

target_column = df_bin.columns[-1]  # Última columna del CSV como la columna objetivo

# Calcular la correlación de todas las columnas con la columna objetivo
correlations = df_bin.corr()[target_column].drop(target_column)  # Excluir la autocorrelación de la columna objetivo
# Ordenar las correlaciones de mayor a menor
correlations_sorted = correlations.sort_values(ascending=True)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=correlations_sorted.values,y=correlations_sorted.index,hue=correlations_sorted.values,palette="coolwarm")  # Azul para correlación negativa, rojo para positiva
plt.title(f"Correlación de atributos con {target_column}")
plt.xlabel("Coeficiente de Correlación")
plt.ylabel("Atributos")
# Añadir una barra de color al lado derecho del gráfico para explicar el significado de los colores
# Crear el mapeo de color y asociarlo con la barra de color
#plt.xlim(-0.1, 1)
norm = mpl.colors.Normalize(vmin=-0.1, vmax=1)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])  # Necesario para que el ScalarMappable sea compatible con colorbar
cbar = plt.colorbar(sm, orientation="vertical", ax=barplot.axes)
cbar.set_label("Nivel de Correlación")
plt.tight_layout()  # Ajuste automático del espacio
plt.show()

# -------------------------------------------
# 4.Imputacion y Normalizacion
# -------------------------------------------
# -------------------------------------------
# 4.1 Identificar columnas con valores faltantes o caracteres no numéricos
# -------------------------------------------
print("\n Revisión de columnas con valores faltantes o datos inválidos:\n")

# Detectar columnas con valores nulos
missing_values = df_bin.isnull().sum()
missing_columns = missing_values[missing_values > 0]

# Detectar caracteres raros (por ejemplo '?') en variables categóricas
strange_chars = (df_bin[categorical_vars] == '?').sum()
strange_columns = strange_chars[strange_chars > 0]

# Imprimir resultados
if missing_columns.empty and strange_columns.empty:
    print(" No se encontraron valores faltantes ni caracteres extraños en el dataset.")
else:
    if not missing_columns.empty:
        print("⚠️ Columnas con valores nulos:\n", missing_columns)
    if not strange_columns.empty:
        print("⚠️ Columnas con caracteres extraños ('?'):\n", strange_columns)

# Convertir los '?' a np.nan para imputar si se encuentran
for col in strange_columns.index:
    df_bin[col] = df_bin[col].replace('?', np.nan)

# -------------------------------------------
# 4.2 Imputar columnas si es necesario
# -------------------------------------------
if not missing_columns.empty or not strange_columns.empty:
    print("\n🔧 Realizando imputación de datos...")

    imputador = SimpleImputer(strategy='most_frequent')  # útil para categóricas
    df_bin[missing_columns.index.tolist() + strange_columns.index.tolist()] = imputador.fit_transform(
        df_bin[missing_columns.index.tolist() + strange_columns.index.tolist()]
    )
    print(" Imputación completada.")
else:
    print(" No se requiere imputación.")

# -------------------------------------------
# 4.3 Normalizar variables numéricas continuas
# -------------------------------------------


# ——————————————
#  Pairplot de variables clave
# ——————————————

print("Pairplot.......")
vars_sel = ['age', 'trestbps', 'chol', 'thalach', 'num_binarizada']
sns.pairplot(df_bin[vars_sel], hue='num_binarizada', corner=True)
plt.show()


# Boxplots específicos
for col in ['oldpeak', 'thalach']:
    plt.figure()
    sns.boxplot(x='num_binarizada', y=col, data=df_bin)
    plt.title(f'Boxplot de {col} por clase')
    plt.show()


print("\n📏 Normalizando variables numéricas continuas...")


# Conteo de outliers por IQR
outliers = {}
numeric_cols = df_bin.select_dtypes(include='number').columns.drop('num_binarizada')
for col in numeric_cols:
    Q1, Q3 = df_bin[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    mask = (df_bin[col] < Q1 - 1.5*IQR) | (df_bin[col] > Q3 + 1.5*IQR)
    outliers[col] = mask.sum()
pd.Series(outliers).sort_values().plot.bar(figsize=(10,4))
plt.title('Número de outliers por variable')
plt.show()
# ——————————————



# Excluir las categóricas y la columna objetivo
features_to_normalize = df_bin.drop(columns=categorical_vars + ['num_binarizada']).columns.tolist()

scaler = MinMaxScaler()
df_bin[features_to_normalize] = scaler.fit_transform(df_bin[features_to_normalize])

### Bar chart de media ± desviación estándar
# Agrupar y calcular media y desviación estándar por clase
stats = df_bin.groupby('num_binarizada')[numeric_cols].agg(['mean','std'])

# Extraer medias y desviaciones estándar de la clase 1
means_class_1 = stats.loc[1, (slice(None), 'mean')]
stds_class_1 = stats.loc[1, (slice(None), 'std')]

# Extraer medias de la clase 0 para calcular diferencia
means_class_0 = stats.loc[0, (slice(None), 'mean')]

# Calcular diferencia de medias
mean_diff = means_class_1.values - means_class_0.values

# Crear un nuevo DataFrame para graficar
plot_df = pd.DataFrame({
    'mean': means_class_1.values,
    'std': stds_class_1.values,
    'mean_diff': mean_diff
}, index=means_class_1.index)

# Ordenar por la diferencia de medias
plot_df.sort_values('mean_diff', inplace=True)

# Graficar
plot_df.plot.bar(y='mean', yerr='std', figsize=(12,6), legend=False)
plt.ylabel('Valor medio')
plt.title('Media (clase 1) ± Desviación estándar')
plt.tight_layout()
plt.show()



print(f" Variables normalizadas: {features_to_normalize}")

# -------------------------------------------
# 4.4 Codificación one-hot para variables categóricas
# -------------------------------------------
print("\n Aplicando One-Hot Encoding a variables categóricas...")

# Convertir las variables categóricas usando one-hot encoding
df_final = pd.get_dummies(df_bin, columns=categorical_vars, drop_first=True)
# Convertir todas las columnas booleanas a 0 y 1
df_final = df_final.astype({col: 'int' for col in df_final.select_dtypes(include='bool').columns})

print(" One-hot encoding completado.")
print(f" Dimensiones del dataset final: {df_final.shape}")
print("\n Columnas finales:")
print(df_final.columns.tolist())

# -------------------------------------------
# 4.5 Visualización previa del dataset final
# -------------------------------------------
print("\n Vista previa del dataset listo para entrenar:")
display(df_final.head())

### Heatmap: correlación original vs preprocesada

fig, axs = plt.subplots(1,2, figsize=(16,6))
sns.heatmap(df.corr(), ax=axs[0], cmap='coolwarm', cbar=False)
axs[0].set_title('Correlación original')
sns.heatmap(df_bin.corr(), ax=axs[1], cmap='coolwarm', cbar=False)
axs[1].set_title('Correlación preprocesada')
plt.show()



# Guardar el dataset preprocesado
output_path = '/content/drive/My Drive/Colab Notebooks/heart_preprocesado.csv'
df_final.to_csv(output_path, index=False)
print(f"\n Dataset final guardado en: {output_path}")

print(f"\n Revisando que la clase objetivo tenga datos balanceados {output_path}")
print(df_final['num_binarizada'].value_counts(normalize=True))
print(df_final['num_binarizada'].value_counts())


# -------------------------------------------
# 5 Fase de entrenamiento
# -------------------------------------------

# Leer el archivo CSV
input_path = '/content/drive/My Drive/Colab Notebooks/heart_preprocesado.csv'
df_train = pd.read_csv(input_path)
df_handle = df_train.copy()


# Separar características (X) y variable objetivo (y)
X = df_handle.drop(columns=['num_binarizada'])
y = df_handle['num_binarizada']

print("✅ Separación realizada. X tiene shape:", X.shape)
print("y contiene:", y.value_counts().to_dict())

# strafity se usa sobre todo cuando no hay un balance 50% y 50% de los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Prueba:", X_test.shape, y_test.shape)

# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# 1. Crear el modelo
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Entrenar el modelo
modelo_rf.fit(X_train, y_train)

# 3. Realizar predicciones
y_pred = modelo_rf.predict(X_test)

# 4. Evaluar resultados
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Reporte de clasificación:")
print(classification_report(y_test, y_pred))


# Importancia de variables del RF
# ——————————————
importances = pd.Series(modelo_rf.feature_importances_, index=X_train.columns)
importances.sort_values().plot.barh(figsize=(8,10))
plt.title('Importancia de variables – Random Forest')
plt.xlabel('Importancia')
plt.show()

# 5. Matriz de confusión


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(' Matriz de Confusión - Random Forest')
plt.show()

##  ROC, AUC

y_probs_rf = modelo_rf.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
auc_rf = roc_auc_score(y_test, y_probs_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal para referencia
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

##################################################################
# modelo Gradient Boosting
modelo_gb = GradientBoostingClassifier(random_state=42)

# Entrenar el modelo
modelo_gb.fit(X_train, y_train)

# Realizar predicciones
y_pred_gb = modelo_gb.predict(X_test)

# Evaluar resultados
print(" Resultados con Gradient Boosting:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_gb))

# Matriz de confusión para Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(' Matriz de Confusión - Gradient Boosting')
plt.show()

### ROC, AUC###
y_probs_gb = modelo_gb.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_probs_gb)
auc_gb = roc_auc_score(y_test, y_probs_gb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='green', label=f'Gradient Boosting (AUC = {auc_gb:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Gradient Boosting')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

########Red neuronal simple#########
# Crear una red neuronal simple
# Modelo de red neuronal simple
nn_model = Sequential()
nn_model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compilar y entrenar
nn_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Predicciones
y_pred_nn_proba = nn_model.predict(X_test)
y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)

# --- Matriz de confusión (gráfica) ---
cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="pink", xticklabels=[0,1], yticklabels=[0,1])
plt.title("Matriz de Confusión - Red Neuronal")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# --- Reporte de clasificación ---
print("\n--- Red Neuronal ---")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_nn))

# --- Curva ROC y AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_nn_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='hotpink', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos', color='deeppink')
plt.ylabel('Tasa de Verdaderos Positivos', color='deeppink')
plt.title('Curva ROC - Red Neuronal', color='deeppink')
plt.legend(loc="lower right", facecolor='white')
plt.grid(color='lightpink')
plt.show()

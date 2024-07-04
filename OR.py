import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Regresión Logística": LogisticRegression(),
    "Bosque Aleatorio": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}


mejor_modelo = None
mejor_precision = 0
resultados = {}

for nombre, modelo in models.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    resultados[nombre] = precision
    if precision > mejor_precision:
        mejor_precision = precision
        mejor_modelo = modelo

print("Mejor modelo:", mejor_modelo)
print("Precisión:", mejor_precision)

y_pred = mejor_modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()

data['DEATH_EVENT_PREDICHO'] = mejor_modelo.predict(scaler.transform(X))

casos_positivos = data[data['DEATH_EVENT_PREDICHO'] == 1]
casos_negativos = data[data['DEATH_EVENT_PREDICHO'] == 0]

plt.figure(figsize=(10,7))
sns.boxplot(x='anaemia', y='age', data=data)
plt.xlabel('Anemia')
plt.ylabel('Edad')
plt.show()
plt.figure(figsize=(10,7))
sns.scatterplot(x='age', y='time', hue='DEATH_EVENT', data=data)
plt.xlabel('Edad')
plt.ylabel('Tiempo')
plt.show()

data.to_csv('heart_failure_predictions.csv', index=False)
casos_positivos.to_csv('heart_failure_positive_cases.csv', index=False)
casos_negativos.to_csv('heart_failure_negative_cases.csv', index=False)

print("Archivos CSV guardados con éxito.")

# Importiranje potrebnih biblioteka
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Učitavanje podataka
data = pd.read_csv("C:/Users/matij/Downloads/archive/Fraud.csv", nrows=1000000)

# Analiza podataka
print(data.info())
print(data.describe())

# Razdvajanje značajki (features) i ciljne varijable (target)
X = data[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = data['isFraud']

# Razdvajanje podataka na skup za učenje i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Izolacija šuma za analizu anomalija
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_iso_forest = iso_forest.fit_predict(X_train)

# Filtriranje prijevara (klasa -1) iz skupa za učenje
X_train_clean = X_train[y_iso_forest == 1]
y_train_clean = y_train[y_iso_forest == 1]

# Neuronska mreža za klasifikaciju
clf = MLPClassifier(hidden_layer_sizes=(64, 64), random_state=42)
clf.fit(X_train_clean, y_train_clean)

# Predikcija na skupu za testiranje
y_pred = clf.predict(X_test)

# Evaluacija performansi modela
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Ispis rezultata
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Vizualizacija matrice konfuzije
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Vizualizacija ROC krivulje
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('SUs.csv')

X = data.iloc[:, 1:]  
y = data.iloc[:, 0]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

param_grid = {'n_neighbors': np.arange(1, 31)}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)


y_pred = grid_search.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False  

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())



ax = plt.gca()  
ax.tick_params(axis='both', labelsize=24)  

yticklabels = ax.get_yticklabels()
xticklabels = ax.get_xticklabels()
for label in yticklabels + xticklabels:
    label.set_fontweight('bold')
    label.set_fontsize(24)  

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.set_yticklabels(y.unique(), fontweight='bold', fontsize=24)
ax.set_xticklabels(y.unique(), fontweight='bold', fontsize=24)

ax.set_ylabel('True Label', fontsize=24, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=24, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_knn_300dpi.png', dpi=300, bbox_inches='tight')
plt.show()

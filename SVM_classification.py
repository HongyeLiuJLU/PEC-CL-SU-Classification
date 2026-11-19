import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('SUs.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train_std, y_train)

y_pred = clf.predict(X_test_std)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"SVC‑RBF Accuracy: {acc:.4f}")
print(f"SVC‑RBF F1‑macro: {f1_macro:.4f}")

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(7, 6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_ylabel('True Label', fontsize=24, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion matrix_SVC_300dpi.png', dpi=300, bbox_inches='tight')
plt.show()

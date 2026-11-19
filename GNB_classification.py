import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

df = pd.read_csv('SUs no C.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

clf = GaussianNB()
clf.fit(X_train_std, y_train)

y_pred = clf.predict(X_test_std)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

print(f"Accuracy: {acc:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"F1 (micro): {f1_micro:.4f}")

plt.figure(figsize=(7, 6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())

ax = plt.gca()
ax.tick_params(axis='both', labelsize=24)

yticklabels = ax.get_yticklabels()
xticklabels = ax.get_xticklabels()
for label in yticklabels + xticklabels:
    label.set_fontweight('bold')
    label.set_fontsize(24)

for spine in ax.spines.values():
    spine.set_linewidth(3)

ax.set_yticklabels(clf.classes_, fontweight='bold', fontsize=24)
ax.set_xticklabels(clf.classes_, fontweight='bold', fontsize=24)
ax.set_ylabel('True Label', fontsize=24, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=24, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion matrix_GNB_300dpi.png', dpi=300, bbox_inches='tight')
plt.show()

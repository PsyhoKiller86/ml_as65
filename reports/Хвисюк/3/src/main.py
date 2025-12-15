import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC


# 1 загрузка стандартиз
# 2 раздел выборки
df = pd.read_csv("glass.csv")
X, y = df.drop("Type", axis=1), df["Type"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 3 обуч
# 4 сравн трёх моделей
best_k, best_acc = max(
    ((k, accuracy_score(y_test, KNeighborsClassifier(k).fit(X_train, y_train).predict(X_test)))
     for k in range(1, 10)),
    key=lambda x: x[1]
)
print(f"The best k: {best_k}, accuracy: {best_acc:.4f}\n")

models = {
    f"k-NN (k={best_k})": KNeighborsClassifier(best_k),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
   
   
    # 5 худший класс
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recalls = {cls: m["recall"] for cls, m in report.items() if cls.isdigit()}
    worst = min(recalls, key=recalls.get)
    print(f"Худший класс — {worst} (recall={recalls[worst]:.2f})\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#   Daten laden
df = pd.read_excel("fruit_data.xlsx")

#   Kategorien umwandeln
encoder = LabelEncoder()
df["color"] = encoder.fit_transform(df["color"])
df["fruit_type"] = encoder.fit_transform(df["fruit_type"])
df["size"] = encoder.fit_transform(df["size"])

#   Eingabe und Ausgabemerkmale festlegen
X = df[["weight","color","size"]]
y = df["fruit_type"]

#   Daten in Trainings- (80%) und Testdaten (20%) aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 35)

#   Random Forest Modell
model = RandomForestClassifier(
    n_estimators = 400,
    max_depth = None,
    min_samples_split = 3,
    min_samples_leaf = 2,
    max_features = 'sqrt',
    bootstrap = True,
    random_state = 7
)

#   Trainieren des Modells
model.fit(X_train, y_train)

#   Testen
y_pred = model.predict(X_test)
print("Genauigkeit:", accuracy_score(y_test, y_pred))
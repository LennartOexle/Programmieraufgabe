import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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

#   One-hot-Encoding für Zielvariable
y = to_categorical(y)

#   Daten in Trainings- (80%) und Testdaten (20%) aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 35)

#   Skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   Modell definieren
model = Sequential([
    Input(shape = (X_train.shape[1],)),
    Dense(32, activation = 'relu'),
    Dropout(0.2),
    Dense(16, activation = 'relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation = 'softmax')
])

#   Kompilieren
model.compile(
    optimizer = Adam(learning_rate = 0.00056),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#   Early Stopping
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    restore_best_weights = True
)

#   Trainieren
history = model.fit(
    X_train, y_train,
    epochs = 200,
    batch_size = 5,
    validation_split = 0.2,
    callbacks = [early_stop],
    verbose = 1
)

# Testen
loss, acc = model.evaluate(X_test, y_test, verbose = 0)
print(f"Genaugkeit: {acc:.3f}")
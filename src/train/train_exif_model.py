import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/metadata.csv")

X = df[["missing","has_exif","suspicious"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8
)

model.fit(X_train, y_train)

joblib.dump(model, "src/models/exif_xgb.pkl")

print("XGBoost EXIF model saved 🚀")
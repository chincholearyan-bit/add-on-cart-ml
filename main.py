import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. LOAD DATA
# ===============================

users = pd.read_csv("users.csv")
restaurants = pd.read_csv("restaurants.csv")
items = pd.read_csv("items.csv")
orders = pd.read_csv("orders.csv")
order_items = pd.read_csv("order_items.csv")

print("Data Loaded Successfully")

# ===============================
# 2. MERGE DATA
# ===============================

# Merge orders with users
df = orders.merge(users, on="user_id")

# Merge with restaurants
df = df.merge(restaurants, on="rest_id")

# Convert datetime
df["order_time"] = pd.to_datetime(df["order_time"])
df["order_hour"] = df["order_time"].dt.hour

print("Orders merged")

# ===============================
# 3. CREATE TARGET VARIABLE
# ===============================

# Merge order_items with items to get category
merged_items = order_items.merge(items,
                                 left_on="items_id",
                                 right_on="item_id")

# Identify orders where drinks were added
drink_orders = merged_items[
    merged_items["category"] == "drinks"
]["order_id"].unique()

# Create binary target
df["added_drink"] = df["order_id"].apply(
    lambda x: 1 if x in drink_orders else 0
)

print("Target variable created")

# ===============================
# 4. ENCODE CATEGORICAL FEATURES
# ===============================

le_price = LabelEncoder()
le_cuisine = LabelEncoder()

df["price_range"] = le_price.fit_transform(df["price_range"])
df["cuisine"] = le_cuisine.fit_transform(df["cuisine"])

# ===============================
# 5. SELECT FEATURES
# ===============================

features = [
    "weekday",
    "weekend_flag",
    "veg_pref",
    "price_sensitivity",
    "order_frequency",
    "rating",
    "order_hour",
    "price_range",
    "cuisine"
]

X = df[features]
y = df["added_drink"]

# ===============================
# 6. TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 7. TRAIN MODEL
# ===============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")

# ===============================
# 8. EVALUATION
# ===============================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 9. FEATURE IMPORTANCE
# ===============================

importances = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importances)

# ===============================
# 10. PREDICT FOR NEW ORDER
# ===============================

sample_order = X_test.iloc[0:1]

prediction = model.predict(sample_order)
probability = model.predict_proba(sample_order)

print("\nPrediction (Drink Added?):", prediction)
print("Probability:", probability)
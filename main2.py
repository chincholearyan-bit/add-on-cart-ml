import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

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

df = orders.merge(users, on="user_id")
df = df.merge(restaurants, on="rest_id")


df["order_time"] = pd.to_datetime(df["order_time"])
df["order_hour"] = df["order_time"].dt.hour

print("Orders merged")

# ===============================
# 3. CREATE TARGET VARIABLE
# ===============================

# Merge order_items with items to get category
merged_items = order_items.merge(items,
                                 left_on="items_id",
                                 right_on="item_id"
                                 )

addon_map = {}

for order_id in merged_items["order_id"].unique():
  
  order_data = merged_items[merged_items["order_id"]==order_id]

  non_main = order_data[order_data["category"] != "main"]

  if len(non_main) > 0:
    addon_map[order_id] = non_main.iloc[0]["category"]
  else:
    addon_map[order_id] = "none"

df["addon_category"] = df["order_id"].map(addon_map)

print("Target column created")

# drink_orders = merged_items[
#     merged_items["category"] == "drinks"
# ]["order_id"].unique()

# # Create binary target
# df["added_drink"] = df["order_id"].apply(
#     lambda x: 1 if x in drink_orders else 0
# )

# print("Target variable created")

# ===============================
# 4. ENCODE CATEGORICAL FEATURES
# ===============================

le_price = LabelEncoder()
le_cuisine = LabelEncoder()
le_target = LabelEncoder()

df["price_range"] = le_price.fit_transform(df["price_range"])
df["cuisine"] = le_cuisine.fit_transform(df["cuisine"])
df["addon_category"] = le_target.fit_transform(df["cuisine"])

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
y = df["addon_category"]

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
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")

# ===============================
# 8. EVALUATION
# ===============================

y_pred = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


joblib.dump(model, "addon_model.pkl")
joblib.dump(le_target, "target_encode.pkl")

print("Model saved")

# ===============================
# 9. SUGGESTION FUNCTION
# ===============================






# importance_df = pd.DataFrame({
#     "Feature": features,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# print("\nFeature Importance:\n")
# print(importance_df)

# ===============================
# 10. PREDICT FOR NEW ORDER
# ===============================

# sample_order = X_test.iloc[0:1]

# prediction = model.predict(sample_order)
# probability = model.predict_proba(sample_order)

# print("\nPrediction (Drink Added?):", prediction)
# print("Probability:", probability)


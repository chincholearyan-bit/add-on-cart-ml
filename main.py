import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib



users = pd.read_csv("users.csv")
restaurants = pd.read_csv("restaurants.csv")
items = pd.read_csv("items.csv")
orders = pd.read_csv("orders.csv")
order_items = pd.read_csv("order_items.csv")

print("Data Loaded Successfully")



df = orders.merge(users, on="user_id")
df = df.merge(restaurants, on="rest_id")


df["order_time"] = pd.to_datetime(df["order_time"])
df["order_hour"] = df["order_time"].dt.hour

print("Orders merged")


merged_items = order_items.merge(items,
                                 left_on="items_id",
                                 right_on="item_id"
                                 )

addon_map = {}
main_map = {}

for order_id in merged_items["order_id"].unique():
  
  order_data = merged_items[merged_items["order_id"]==order_id]

  main_item = order_data[order_data["category"] == "main"]

  if len(main_item) > 0:
    main_map[order_id] = main_item.iloc[0]["food_type"]
  else:
    main_map[order_id] = "unknown"

  non_main = order_data[order_data["category"] != "main"]

  if len(non_main) > 0:
    addon_map[order_id] = non_main.iloc[0]["category"]
  else:
    addon_map[order_id] = "none"

df["main_food_type"] = df["order_id"].map(main_map)
df["addon_category"] = df["order_id"].map(addon_map)

print("Target column created")


le_price = LabelEncoder()
le_cuisine = LabelEncoder()
le_target = LabelEncoder()
le_main = LabelEncoder()

df["price_range"] = le_price.fit_transform(df["price_range"])
df["cuisine"] = le_cuisine.fit_transform(df["cuisine"])
df["addon_category"] = le_target.fit_transform(df["addon_category"])
df["main_food_type"] = le_main.fit_transform(df["main_food_type"])


features = [
    "weekday",
    "weekend_flag",
    "veg_pref",
    "price_sensitivity",
    "order_frequency",
    "rating",
    "order_hour",
    "price_range",
    "cuisine",
    "main_food_type"
]

X = df[features]
y = df["addon_category"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


joblib.dump(model, "addon_model.pkl")
joblib.dump(le_target, "target_encoder.pkl")
joblib.dump(le_main, "main_encoder.pkl")

print("Model saved")


def suggest_addon(user_input_dict, rest_id, main_food_type_input):
  new_data = pd.DataFrame([user_input_dict])

  # new_data["price_range"] = le_price.transform(new_data["price_range"])
  # new_data["cuisine"] = le_cuisine.transform(new_data["cuisine"])
  new_data["main_food_type"] = le_main.transform([main_food_type_input])
  
  probs = model.predict_proba(new_data)[0]

  # category_names = le_target.classes_

  category_probs = dict(zip(le_target.classes_, probs))

  sorted_categories = sorted(
    category_probs.items(),
    key=lambda x: x[1],
    reverse=True

  )

  print("\nPredicted Probablities:")
  for cat, prob in sorted_categories:
    print(f"{cat}: {round(prob, 3)}")

  # top_categories = [
  #   cat for cat, prob in sorted_categories
  #   if cat != "none"
  # ][:2]

  filtered_categories = [
    cat for cat, prob in sorted_categories
    if cat not in ["none", "main"]
  ]
  
  priority = []

  if main_food_type_input == "curry":
    print("\nBussiness Rule: Priortizing CARBS")
    priority = ["carbs"]

  elif main_food_type_input == "biryani":
    print("\nBussiness Rule: Prioritizing Raita & Drinks")
    priority = ["sides", "drinks", "dessert"]

  # else:
  #   priority = []

  final_categories = priority + [
  cat for cat in filtered_categories
  if cat not in priority
  ]
  final_categories = final_categories[:2]

  print("\nTop Suggested Categories:", final_categories)
 
  for category in final_categories:

    # items_in_category = items[
    #   (items["rest_id"] == rest_id)&
    #   (items["category"] == category)
    # ]
    if category == "carbs":
      items_in_category = items[
        (items["rest_id"] == rest_id) &
        (items["food_type"] == "carbs")
    ]
    else:
      items_in_category = items[
        (items["rest_id"] == rest_id) &
        (items["category"] == category)
    ]
    print(f"\nSuggested {category} items: ")

    for item in items_in_category["item_name"].head(3):
      print("-",item)


print("\n=== Simulating New Order ===")

example_input = {
  "weekday": 2,
  "weekend_flag": 0,
  "veg_pref": 1,
  "price_sensitivity": 0.3,
  "order_frequency": 5,
  "rating": 4.2,
  "order_hour": 21,
  "price_range": le_price.transform(["Medium"])[0],
  "cuisine": le_cuisine.transform(["North Indian"])[0]
}

suggest_addon(
  example_input, 
  rest_id=10,
  main_food_type_input="curry"
  )





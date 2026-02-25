import pandas as pd
import numpy as np
from datetime import datetime, timedelta


np.random.seed(42)

n_users = 3000

# cities = ["Mumbai", "Pune", "Delhi","Nagpur"]

users_df = pd.DataFrame({
  "user_id": range(1, n_users+1),
  "veg_pref": np.random.choice([0,1],n_users, p=[0.6,0.4]),
  "price_sensitivity": np.random.uniform(0,1,n_users),
  "order_frequency": np.random.poisson(6, n_users)
})

n_rest = 120

cuisines = ["North Indian","South Indian","Chinese","Italian","Fast Food"]

restaurants_df = pd.DataFrame({
  "rest_id": range(1,n_rest+1),
  "cuisine": np.random.choice(cuisines,n_rest),
  "price_range": np.random.choice(["Low", "Medium", "High"],n_rest),
  "rating": np.round(np.random.uniform(3.5,5.0,n_rest),2)
})

main_items = [
    ("Butter Chicken", "curry"),
    ("Paneer Curry", "curry"),
    ("Dal Tadka", "curry"),
    ("Veg Biryani", "biryani"),
    ("Chicken Biryani", "biryani"),
    ("Dal Khichadi","biryani"),
    ("Chapati", "carbs"),
    ("Tandori","carbs"),
    ("Nans","carbs"),
    ("Garlic Nan","carbs"),
    ("Jeera Rice", "carbs"),
    ("Garlic Rice","carbs"),
    ("Plain Rice", "carbs")
]

drink_items = ["Coke","Pepsi","Lassi","Cold Coffee","Buttermilk"]
dessert_items = ["Gulab Jamun","RasMalai","Rasgulla","Jalebi","Ice Cream"]
side_items = ["Raita","Salad","Roasted Papad"]


# categories = ["main", "drinks", "dessert", "sides"]

items_list = []
item_id = 1

for _, row in restaurants_df.iterrows():

  for item_name, food_type in main_items:
    items_list.append([
      item_id,
      row["rest_id"],
      item_name,
      "main",
      food_type,
      round(np.random.uniform(150, 400), 2),
      np.random.choice([0, 1], p=[0.4, 0.6])
    ])
    item_id +=1

  for item in drink_items:
    items_list.append([
        item_id,
        row["rest_id"],
        item,
        "drinks",
        "beverage",
        round(np.random.uniform(40, 150), 2),
        1
    ])
    item_id += 1

  for item in dessert_items:
    items_list.append([
      item_id,
      row["rest_id"],
      item,
      "dessert",
      "sweet",
      round(np.random.uniform(80, 200), 2),
      1
    ])
    item_id +=1

  for item in side_items:
    items_list.append([
      item_id,
      row["rest_id"],
      item,
      "sides",
      "addon",
      round(np.random.uniform(40, 120), 2),
      1
    ])
    item_id += 1
  
items_df = pd.DataFrame(items_list, columns=[
    "item_id", "rest_id", "item_name",
    "category", "food_type", "price", "veg_flag"
  ])
  # for cat in categories:
#     for i in range(5):
#       price = np.random.uniform(100,500) if cat=="main" else np.random.uniform(50,200)

#       items_list.append([
#         item_id,
#         row["rest_id"],
#         f"{cat}_item_{i}_{row['rest_id']}",
#         cat,
#         round(price,2),
#         np.random.choice([0,1],p=[0.4,0.6])
#       ])
#       item_id +=1
# items_df = pd.DataFrame(items_list, columns=[
#   "item_id","rest_id","item_name","category","price","veg_flag"
# ])

n_orders = 20000

start_date = datetime(2024,1,1)

orders_list = []

for i in range(1,n_orders+1):
  user = users_df.sample(1).iloc[0]
  rest = restaurants_df.sample(1).iloc[0]
  order_time = start_date + timedelta(days=np.random.randint(0,180),
                                      hours=np.random.randint(0,24)
                                      )
  
  orders_list.append([
    i,
    user["user_id"],
    rest["rest_id"],
    order_time,
    order_time.weekday(),
    1 if order_time.weekday()>=5 else 0
  ])

order_df = pd.DataFrame(orders_list, columns=[
  "order_id","user_id","rest_id","order_time","weekday","weekend_flag"
])


order_items_list = []

for _, order in order_df.iterrows():
  # user = users_df[users_df.user_id==order.user_id].iloc[0]
  rest_items = items_df[items_df.rest_id==order.rest_id]

  main_item = rest_items[rest_items.category=="main"].sample(1)
  order_items_list.append([order.order_id, main_item.item_id.values[0], 1])

  main_food_type = main_item.food_type.values[0]

  hour = order.order_time.hour

  if main_food_type == "curry" and np.random.rand() < 0.7:
    carbs = rest_items[rest_items.food_type == "carbs"].sample(1)
    order_items_list.append([order.order_id, carbs.item_id.values[0], 1])

  if main_food_type == "biryani" and np.random.rand() < 0.6:
    raita = rest_items[rest_items.item_name == "Raita"]
    if len(raita) > 0:
      order_items_list.append([order.order_id, raita.item_id.values[0], 1])


  drink_prob = 0.4
  if hour >= 18:
    drink_prob += 0.1

  available_drinks = rest_items[rest_items.category == "drinks"]

  if len(available_drinks) > 0 and np.random.rand() < drink_prob:
    drink = available_drinks.sample(1)
    order_items_list.append([
        order.order_id,
        drink.item_id.values[0],
        1
    ])

  dessert_prob = 0.25

  if order.weekend_flag == 1:
    dessert_prob += 0.2

  available_dessert = rest_items[rest_items.category == "dessert"]

  if len(available_dessert) > 0 and np.random.rand() < dessert_prob:
    dessert = available_dessert.sample(1)
    order_items_list.append([
        order.order_id,
        dessert.item_id.values[0],
        1
    ])

  if main_food_type == "curry":
    side_prob = 0.6
  elif main_food_type == "biryani":
    side_prob = 0.7
  else:
    side_prob = 0.3


  available_sides = rest_items[rest_items.category == "sides"]

  if len(available_sides) > 0 and np.random.rand() < side_prob:
    side = available_sides.sample(1)
    order_items_list.append([
        order.order_id,
        side.item_id.values[0],
        1
    ])


order_items_df = pd.DataFrame(order_items_list,
                              columns=["order_id","items_id","is_added"])


users_df.to_csv("users.csv", index=False)
restaurants_df.to_csv("restaurants.csv", index=False)
items_df.to_csv("items.csv", index=False)
order_df.to_csv("orders.csv", index=False)
order_items_df.to_csv("order_items.csv", index=False)

print("Dataset generated successfully!")
print(order_items_df.shape)
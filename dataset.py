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

categories = ["main", "drinks", "dessert", "sides"]

items_list = []
item_id = 1

for _, row in restaurants_df.iterrows():
  for cat in categories:
    for i in range(5):
      price = np.random.uniform(100,500) if cat=="main" else np.random.uniform(50,200)

      items_list.append([
        item_id,
        row["rest_id"],
        f"{cat}_item_{i}_{row['rest_id']}",
        cat,
        round(price,2),
        np.random.choice([0,1],p=[0.4,0.6])
      ])
      item_id +=1
items_df = pd.DataFrame(items_list, columns=[
  "item_id","rest_id","item_name","category","price","veg_flag"
])

n_orders = 20000

start_date = datetime(2024,1,1)

orders_list = []

for i in range(1,n_orders+1):
  user = users_df.sample(1).iloc[0]
  rest = restaurants_df.sample(1).iloc[0]
  order_time = start_date + timedelta(days=np.random.randint(0,180),
                                      hours=np.random.randint(0,24))
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
  user = users_df[users_df.user_id==order.user_id].iloc[0]
  rest_items = items_df[items_df.rest_id==order.rest_id]

  main_item = rest_items[rest_items.category=="main"].sample(1)
  order_items_list.append([order.order_id, main_item.item_id.values[0], 1])

  hour = order.order_time.hour

  drink_prob = 0.4
  if hour >= 18: drink_prob +=0.1
  if np.random.rand() < drink_prob:
    drink = rest_items[rest_items.category=="drinks"].sample(1)
    order_items_list.append([order.order_id, drink.item_id.values[0], 1])

  dessert_prob = 0.25
  if order.weekend_flag == 1: dessert_prob += 0.2
  if np.random.rand() < dessert_prob:
    dessert = rest_items[rest_items.category=="dessert"].sample(1)
    order_items_list.append([order.order_id, dessert.item_id.values[0], 1])


  if np.random.rand() < 0.3:
    side = rest_items[rest_items.category=="sides"].sample(1)
    order_items_list.append([order.order_id, side.item_id.values[0], 1])

order_items_df = pd.DataFrame(order_items_list,
                              columns=["order_id","items_id","is_added"])


users_df.to_csv("users.csv", index=False)
restaurants_df.to_csv("restaurants.csv", index=False)
items_df.to_csv("items.csv", index=False)
order_df.to_csv("orders.csv", index=False)
order_items_df.to_csv("order_items.csv", index=False)

print("Dataset generated successfully!")
print(order_items_df.shape)
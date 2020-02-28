import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from olist.data import Olist
from olist.order import Order

model = 'random_forest_3'

print("Importing dataset")

orders = Order().get_training_data()
data = Olist().get_data()

products = data['olist_products_dataset']
order_items = data['olist_order_items_dataset']

# Build training set
order_items_products = \
    order_items.groupby(['order_id',
                         'product_id'],
                        as_index=False)\
               .agg({'order_item_id': 'count'})\
               .merge(products.drop(['product_category_name'],
                                    axis=1),
                      on='product_id')

order_items_products = order_items_products\
                        .groupby('order_id',
                                 as_index=False)\
                        .agg({'product_weight_g': 'sum',
                              'product_length_cm': 'max',
                              'product_height_cm': 'max',
                              'product_width_cm': 'max'})

orders_2 = orders.merge(order_items_products,
                        on='order_id',
                        how='left').dropna()

# Create train and target variable
X_2 = orders_2.drop(['order_id', 'expected_wait_time',
                     'delay_vs_expected', 'wait_time'], axis=1)
y_2 = orders_2['wait_time']

# Training test split
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(
    X_2, y_2, test_size=0.30, random_state=42)

# Average CV score on the training set was:-62.99467662836531
exported_pipeline = RandomForestRegressor(bootstrap=True,
                                          max_features=0.2,
                                          min_samples_leaf=4,
                                          min_samples_split=15,
                                          n_estimators=100)
exported_pipeline.fit(X_2_train,
                      y_2_train)
results = exported_pipeline.predict(X_2_test)

mse = mean_squared_error(y_true=y_2_test,
                         y_pred=results)
print("model: "+model)
print("mean square error:"+str(mse))

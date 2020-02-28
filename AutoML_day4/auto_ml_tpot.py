import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

#import data
from olist.order import Order
from olist.data import Olist

data = Olist().get_data()
training_orders = Order().get_training_data()

orders = data['olist_orders_dataset']

orders['estimate_wait_time'] = (pd.to_datetime(orders['order_estimated_delivery_date'])\
    - pd.to_datetime(orders['order_purchase_timestamp'])) / np.timedelta64(24, 'h')

training_orders =\
    training_orders.merge(orders[['estimate_wait_time', 'order_id']], on='order_id')

X = training_orders.drop(['order_id', 'wait_time', 'delay_vs_expected'], axis=1)
y= training_orders['wait_time']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
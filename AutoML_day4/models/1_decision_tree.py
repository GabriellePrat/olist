import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from olist.order import Order

model = 'decision_tree_1'

print("Importing dataset")
orders = Order().get_training_data()

# Create train and target variable
X = orders.drop(['order_id', 'wait_time', 'delay_vs_expected',
                 'expected_wait_time'], axis=1)
y = orders['wait_time']

# Training test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

# Average CV score on the training set was:-62.99467662836531
exported_pipeline = DecisionTreeRegressor(max_depth=9,
                                          min_samples_leaf=18,
                                          min_samples_split=10)

exported_pipeline.fit(X_train,
                      y_train)
results = exported_pipeline.predict(X_test)

mse = mean_squared_error(y_true=y_test,
                                         y_pred=results)
print("model: "+model)
print("mean square error:"+str(mse))

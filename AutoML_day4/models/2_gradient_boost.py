import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from olist.order import Order

model = 'gradient_boost_2'

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
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    GradientBoostingRegressor(alpha=0.99,
                              learning_rate=0.1,
                              loss="huber",
                              max_depth=8,
                              max_features=0.05,
                              min_samples_leaf=20,
                              min_samples_split=13,
                              n_estimators=100,
                              subsample=0.6000000000000001)
)

exported_pipeline.fit(X_train,
                      y_train)
results = exported_pipeline.predict(X_test)

mse = mean_squared_error(y_true=y_test,
                         y_pred=results)
print("model: "+model)
print("mean square error:"+str(mse))

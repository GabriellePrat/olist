import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, wait_time, expected_wait_time ,delay_vs_expected
        """
        orders = self.data['olist_orders_dataset']
        orders = orders.query("order_status=='delivered'").reset_index()
        orders = orders.dropna()
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] -\
                               orders['order_purchase_timestamp']).astype('timedelta64[D]')
        orders['wait_time'] = (orders['order_delivered_customer_date'] -\
                               orders['order_purchase_timestamp']).astype('timedelta64[D]')
        orders['delay_vs_expected'] = (orders['wait_time'] -\
                                       orders['expected_wait_time'])
        orders['delay_vs_expected'] = orders.delay_vs_expected.apply(lambda row: row if row>0 else 0)
        orders = orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected']]
        return orders

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['olist_order_reviews_dataset']
        reviews.loc[reviews['review_score'] == 5, 'dim_is_five_star'] = 1
        reviews.loc[reviews['review_score'] != 5, 'dim_is_five_star'] = 0
        reviews.loc[reviews['review_score'] == 1, 'dim_is_one_star'] = 1
        reviews.loc[reviews['review_score'] != 1, 'dim_is_one_star'] = 0
        reviews = reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]
        return reviews


    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """
        return pd.DataFrame(self.data['olist_order_items_dataset']\
            .groupby('order_id').product_id.count())\
            .reset_index()\
            .rename(columns={"product_id": "number_of_products"})

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """
        return pd.DataFrame(\
            self.data['olist_order_items_dataset'].groupby('order_id').seller_id.count())\
            .reset_index()\
            .rename(columns={"seller_id": "number_of_sellers"})

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """
        price_and_freight = self.data['olist_order_items_dataset']
        price_and_freight = price_and_freight[['order_id', 'price', 'freight_value']]
        return price_and_freight

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """
        geo = self.data['olist_geolocation_dataset'].groupby('geolocation_zip_code_prefix', as_index=False).first()
        customers = self.data['olist_customers_dataset']
        sellers = self.data['olist_sellers_dataset']
        orders = self.data['olist_orders_dataset']
        orders_items = self.data['olist_order_items_dataset']
        distances = orders.merge(orders_items, how='left', on='order_id')
        distances = distances.merge(customers, how='left', on='customer_id')
        distances = distances.merge(sellers, how='left', on='seller_id')
        distances = distances[['order_id', 'customer_zip_code_prefix', 'seller_zip_code_prefix']]
        distances = distances.merge(geo, how='left', left_on='customer_zip_code_prefix', \
                           right_on='geolocation_zip_code_prefix')
        distances = distances.merge(geo, how='left', left_on='seller_zip_code_prefix', \
                           right_on='geolocation_zip_code_prefix', suffixes=('_customer', '_seller'))
        distances['distances'] = distances.apply(lambda x: haversine_distance(x['geolocation_lng_customer'], \
                                                                     x['geolocation_lat_customer'], \
                                                                     x['geolocation_lng_seller'], \
                                                                     x['geolocation_lat_seller']), axis=1)
        distances = distances[['order_id', 'distances']]
        return distances
        # Optional
        # Hint: you can use the haversine_distance logic coded in olist/utils.py

    def get_training_data(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, wait_time, wait_vs_expected,
        dim_is_five_star, dim_is_one_star, number_of_product,
        number_of_sellers, freight_value, distance_customer_seller
        """
        df = self.get_wait_time().merge(self.get_review_score(), how='outer')\
            .merge(self.get_number_products(), how="outer")\
            .merge(self.get_number_sellers(), how="outer")\
            .merge(self.get_price_and_freight(), how="outer")\
            .merge(self.get_distance_seller_customer(), how='outer')
        return df




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

        # filter only delivered orders
        orders = self.data['olist_orders_dataset'].query("order_status=='delivered'")
        orders = orders.dropna()

        # handle datetime
        orders['order_delivered_customer_date'] = \
            pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = \
            pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['order_purchase_timestamp'] = \
            pd.to_datetime(orders['order_purchase_timestamp'])

        # compute delay vs expected
        orders['delay_vs_expected'] = \
            (orders['order_estimated_delivery_date'] -
             orders['order_delivered_customer_date']) / np.timedelta64(24, 'h')

        def handle_delay(x):
            if x < 0:
                return abs(x)
            else:
                return 0

        orders['delay_vs_expected'] = \
            orders['delay_vs_expected'].apply(handle_delay)

        # compute wait time
        orders['wait_time'] = \
            (orders['order_delivered_customer_date'] -
             orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        # compute wait time
        orders['expected_wait_time'] = \
            (orders['order_estimated_delivery_date'] -
             orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

        return orders[['order_id', 'wait_time', 'expected_wait_time',
                       'delay_vs_expected']]

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star
        """

        # import data
        reviews = self.data['olist_order_reviews_dataset']

        def dim_five_star(d):
            if d == 5:
                return 1
            else:
                return 0

        def dim_one_star(d):
            if d == 1:
                return 1
            else:
                return 0

        reviews['dim_is_five_star'] =\
            reviews['review_score'].apply(dim_five_star)

        reviews['dim_is_one_star'] =\
            reviews['review_score'].apply(dim_one_star)

        return reviews[['order_id', 'dim_is_five_star',
                        'dim_is_one_star', 'review_score']]

    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """

        data = self.data
        products = \
            data['olist_order_items_dataset']\
            .groupby('order_id',
                     as_index=False).agg({'order_item_id': 'max'})
        products.columns = ['order_id', 'number_of_products']
        return products

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """

        data = self.data
        sellers = \
            data['olist_order_items_dataset']\
            .groupby('order_id')['seller_id'].nunique().reset_index()
        sellers.columns = ['order_id', 'number_of_sellers']

        return sellers

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """

        data = self.data
        price_freight = \
            data['olist_order_items_dataset']\
            .groupby('order_id',
                     as_index=False).agg({'price': 'sum',
                                          'freight_value': 'sum'})

        return price_freight

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """

        # import data

        data = self.data
        matching_table = Olist().get_matching_table()

        # Since one zipcode can map to multiple (lat, lng), take first one
        geo = data['olist_geolocation_dataset']
        geo = geo.groupby('geolocation_zip_code_prefix',
                          as_index=False).first()

        # Select sellers and customers
        sellers = data['olist_sellers_dataset']
        customers = data['olist_customers_dataset']

        # Merge geo_location for sellers
        sellers_mask_columns = ['seller_id', 'seller_zip_code_prefix',
                                'seller_city', 'seller_state',
                                'geolocation_lat', 'geolocation_lng']

        sellers_geo = sellers.merge(geo,
                                    how='left',
                                    left_on='seller_zip_code_prefix',
                                    right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        # Merge geo_location for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix',
                                  'customer_city', 'customer_state',
                                  'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(geo,
                                        how='left',
                                        left_on='customer_zip_code_prefix',
                                        right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        # Use the matching table and merge customers and sellers
        matching_geo = matching_table.merge(sellers_geo,
                                            on='seller_id')\
                                     .merge(customers_geo,
                                            on='customer_id',
                                            suffixes=('_seller',
                                                      '_customer'))
        # Remove na()
        matching_geo = matching_geo.dropna()

        matching_geo['distance_seller_customer'] =\
            matching_geo.apply(lambda row:
                               haversine_distance(row['geolocation_lng_seller'],
                                                  row['geolocation_lat_seller'],
                                                  row['geolocation_lng_customer'],
                                                  row['geolocation_lat_customer']),
                               axis=1)
        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance =\
            matching_geo.groupby('order_id',
                                 as_index=False).agg({'distance_seller_customer':
                                                      'mean'})

        return order_distance

    def get_training_data(self):

        training_set =\
            self.get_wait_time()\
                .merge(
                self.get_review_score(), on='order_id'
               ).merge(
                self.get_number_products(), on='order_id'
               ).merge(
                self.get_number_sellers(), on='order_id'
               ).merge(
                self.get_price_and_freight(), on='order_id'
               ).merge(
                self.get_distance_seller_customer(), on='order_id'
               )
        return training_set

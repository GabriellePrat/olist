import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_seller_features(self):
        """
        03-01 > Returns a DataFrame with:
       'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['olist_sellers_dataset']
        sellers.drop('seller_zip_code_prefix',
                     axis=1,
                     inplace=True)
        # There is multiple rows per seller
        sellers.drop_duplicates(inplace=True)

        return sellers

    def get_seller_delay_wait_time(self):
        """
        03-01 > Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """
        # Get data
        order_items = self.data['olist_order_items_dataset']
        orders = self.data['olist_orders_dataset']\
                     .query("order_status=='delivered'")

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship['shipping_limit_date'] =\
            pd.to_datetime(ship['shipping_limit_date'])
        ship['order_delivered_carrier_date'] =\
            pd.to_datetime(ship['order_delivered_carrier_date'])
        ship['order_delivered_customer_date'] =\
            pd.to_datetime(ship['order_delivered_customer_date'])
        ship['order_purchase_timestamp'] =\
            pd.to_datetime(ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean((d.shipping_limit_date
                            - d.order_delivered_carrier_date)/np.timedelta64(24, 'h'))
            if days < 0:
                return abs(days)
            else:
                return 0

        def order_wait_time(d):
            days = np.mean((d.order_delivered_customer_date
                            - d.order_purchase_timestamp)/np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_review_score(self):
        """
        03-01 > Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 
        'review_score'
        """
        matching_table = self.matching_table
        orders_reviews = self.order.get_review_score()

        # Since same seller can appear multiple times in the same
        # order, create a seller <> order matching table

        matching_table = matching_table[['order_id',
                                         'seller_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews,
                                  on='order_id')

        df = df.groupby('seller_id',
                        as_index=False).agg({'dim_is_one_star': 'mean',
                                             'dim_is_five_star': 'mean',
                                             'review_score': 'mean'})
        df.columns = ['seller_id', 'share_one_stars',
                      'share_of_five_stars', 'avg_review_score']

        return df

    def get_quantity(self):
        """
        03-01 > Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity'
        """
        matching_table = self.matching_table

        n_orders =\
            matching_table.groupby('seller_id')['order_id']\
                          .nunique()\
                          .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = \
            matching_table.groupby('seller_id',
                                   as_index=False).agg({'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        return n_orders.merge(quantity, on='seller_id')

    def get_training_data(self):

        training_set =\
            self.get_seller_features()\
                .merge(
                self.get_seller_delay_wait_time(), on='seller_id'
               ).merge(
                self.get_review_score(), on='seller_id'
               ).merge(
                self.get_quantity(), on='seller_id'
               )

        return training_set


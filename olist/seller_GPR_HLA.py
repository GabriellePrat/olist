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
        sellers = sellers.dropna()
        df = sellers[['seller_id', 'seller_city', 'seller_state']]
        return df
    def get_seller_delay_wait_time(self):
        """
        03-01 > Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """
        df_merge_col = pd.merge(self.data['olist_order_items_dataset'], self.data['olist_orders_dataset'], on='order_id')
        # handle datetime
        df_merge_col['shipping_limit_date'] = pd.to_datetime(df_merge_col['shipping_limit_date'])
        df_merge_col['order_purchase_timestamp'] = pd.to_datetime(df_merge_col['order_purchase_timestamp'])
        df_merge_col['order_approved_at'] = pd.to_datetime(df_merge_col['order_approved_at'])
        df_merge_col['order_delivered_carrier_date'] = pd.to_datetime(df_merge_col['order_delivered_carrier_date'])
        df_merge_col['order_estimated_delivery_date'] = pd.to_datetime(df_merge_col['order_estimated_delivery_date'])
        df_merge_col['order_delivered_customer_date'] = pd.to_datetime(df_merge_col['order_delivered_customer_date'])
        # compute delay_to_carrier (if the order is delivered after the shipping limit date,
        # return the number of days between two dates, otherwise 0)
        df_merge_col['delay_to_carrier'] = \
                (df_merge_col['order_delivered_carrier_date']-df_merge_col['shipping_limit_date']) / np.timedelta64(24, 'h')
        #df_merge_col['delay_to_carrier'] = df_merge_col['delay_to_carrier'] if df_merge_col['delay_to_carrier'] > 0 else 0
        # compute seller_wait_time (Average number of days customers waited)
        df_merge_col['seller_wait_time'] = \
                (df_merge_col['order_delivered_customer_date'] - df_merge_col['order_purchase_timestamp']) / np.timedelta64(24, 'h')
        df = df_merge_col.groupby(['seller_id', 'order_id'])[['delay_to_carrier','seller_wait_time']].mean().reset_index()
        df = df_merge_col.groupby(['seller_id'])[['delay_to_carrier','seller_wait_time']].mean().reset_index()
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
        matching_table = matching_table[['order_id','seller_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews, on='order_id')
        df = df.groupby('seller_id', as_index=False).agg({'dim_is_one_star': 'mean',
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
        # n_orders (int) The number of orders the seller was involved with.
        matching_table = self.matching_table
        n_orders = matching_table.groupby('seller_id')['order_id'].nunique().reset_index()
        n_orders = n_orders.rename(columns={'order_id': 'n_orders'})
        quantity = matching_table.groupby('seller_id', as_index=False)['order_id'].count()
        quantity = quantity.rename(columns={'order_id': 'quantity'})
        get_quantity = n_orders.merge(quantity, how='left')
        return get_quantity
    def get_training_data(self):
        """
        03 > 01 Returns a DataFrame with:
        seller_id, seller_state, seller_city, delay_to_carrier,
        seller_wait_time, share_of_five_stars, share_of_one_stars,
        seller_review_score, n_orders
        """
        seller_training_set =\
            self.get_seller_features()\
                .merge(
            self.get_seller_delay_wait_time(), on='seller_id'
               ).merge(
            self.get_review_score(), on='seller_id'
               ).merge(
            self.get_quantity(), on='seller_id'
            )
        return seller_training_set

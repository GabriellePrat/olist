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

    def get_seller_delay_wait_time(self):
        """
        03-01 > Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """

    def get_review_score(self):
        """
        03-01 > Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """

    def get_quantity(self):
        """
        03-01 > Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity'
        """

    def get_training_data(self):
        """
        03 > 01 Returns a DataFrame with:
        seller_id, seller_state, seller_city, delay_to_carrier,
        seller_wait_time, share_of_five_stars, share_of_one_stars,
        seller_review_score, n_orders
        """



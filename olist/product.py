import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Product:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_product_features(self):
        """
        03-01 > Returns a DataFrame with:
       'product_id', 'product_category_name', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm'
        """

    def get_wait_time(self):
        """
        03-01 > Returns a DataFrame with:
        'product_id', 'wait_time'
        """

    def get_review_score(self):
        """
        03-01 > Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """

    def get_quantity(self):
        """
        03-01 > Returns a DataFrame with:
        'product_id', 'n_orders', 'quantity'
        """

    def get_training_data(self):
        """
        03-01 > Returns a DataFrame with:
        product_id, category, height, width, length,
        weight, price, freight_value, product_name_length,
        product_description_length, n_orders, quantity, wait_time,
        share_of_five_stars, share_of_one_stars, review_score
        """

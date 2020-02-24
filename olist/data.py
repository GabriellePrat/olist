import os
import pandas as pd


class Olist:

    def get_data(self):
        """
        01-01 > This function returns all Olist datasets
        as DataFrames within a Python dict.
        """
        # Hint: You will need to find the absolute path of the csv folder in order to call this method from anywhere.
        # Hint 2: look at python __file__ attribute
        data_dict = {}
        abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        for file in os.listdir('{}/data/csv'.format(abs_path)):
            data_dict[file[:-4]] = pd.read_csv('{}/data/csv/{}'.format(abs_path, file))
        return data_dict

if __name__ == '__main__':
    olist = Olist()
    data_dict = olist.get_data()
    print(data_dict.keys())
    print(os.path.abspath(os.path.dirname(__file__)))

    def get_matching_table(self):
        """
        01-01 > This function returns a matching table between
        columns [`customer_id`, `customer_unique_id`,
        `order_id`, `seller_id`]
        """

    def ping(self):
        """
        You call ping I print pong.
        """
        print('pong')

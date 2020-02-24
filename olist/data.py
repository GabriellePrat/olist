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

    def get_matching_table(self):
        """
        01-01 > This function returns a matching table between
        columns [`customer_id`, `customer_unique_id`,
        `order_id`, `seller_id`]
        """
        list_col = ['customer_id', 'customer_unique_id', \
            'order_id', 'product_id', 'seller_id']
        data_dict = self.get_data()
        orders_delivered_df = data_dict['olist_orders_dataset']\
            [data_dict['olist_orders_dataset'].order_status == 'delivered']
        merged_df = orders_delivered_df.merge(right=data_dict['olist_customers_dataset'],
                          how='left',
                          on="customer_id"
                         )
        merged_df = merged_df.merge(right=data_dict['olist_order_items_dataset'],
                          how='left',
                          on="order_id")
        merged_df = data_dict['olist_products_dataset'].merge(right=merged_df,
                          how='left',
                          on="product_id")
        merged_df = data_dict['olist_sellers_dataset'].merge(right=merged_df,
                          how='left',
                          on="seller_id")
        final_df = merged_df[list_col]
        return final_df

    def ping(self):
        """
        You call ping I print pong.
        """
        print('pong')

if __name__ == '__main__':
    olist = Olist()
    print(olist.get_matching_table())

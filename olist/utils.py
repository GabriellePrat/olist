from math import radians, cos, sin, asin, sqrt

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def text_scatterplot(df, x, y):
    """
    For a Dataframe df, create a scatterplot with the index
    as text label.
    """
    f, ax = plt.subplots(figsize=(15, 10))

    f = sns.scatterplot(x=x,
                        y=y,
                        data=df,
                        ax=ax)

    # For each point, we add a text inside the bubble
    for i in df.index:
        f.text(df.loc[i][x],
               df.loc[i][y],
               i,
               horizontalalignment='left')

def return_significative_coef(model):
    """
    Returns p_value, lower and upper bound coefficients
    from a statsmodels object.
    """
    # Extract p_values
    p_values = model.pvalues.reset_index()
    p_values.columns = ['variable', 'p_value']

    # Extract coef_int
    coef = model.params.reset_index()
    coef.columns = ['variable', 'coef']
    return p_values.merge(coef,
                          on='variable')\
                   .query("p_value<0.05").sort_values(by='coef',
                                                      ascending=False)

def plot_kde_plot(df, variable, dimension):
    """
    Plot a side by side kdeplot for `variable`, split
    by `dimension`.
    """
    g = sns.FacetGrid(df,
                      hue=dimension,
                      col=dimension)
    g.map(sns.kdeplot, variable)

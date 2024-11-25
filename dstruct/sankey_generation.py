'''
Author: Nick Usich
Section: Advanced Programming with Data Section 1
Description:
'''

import pandas as pd
import warnings

import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Suppressing Unwanted warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=pd.errors.SettingWithCopyWarning)

def unique_dataframe_mapping(df, src, targ, labels):
    """

    :param df:
    :param src:
    :param targ:
    :param labels:
    :return:
    """
    # Maps the label names from a dataframe into unique numbers for sankey
    # targets and sources

    # makes dataframe of unique targets and values
    comb_labels = pd.concat([df[src], df[targ]]).unique()

    # Appends unique labels to a list for visualization
    for lbl in comb_labels:
        if lbl not in labels:
            labels.append(lbl)

    # Makes a label id for each label
    labelid = {lbl: id for id, lbl in enumerate(labels)}

    # Maps the label id to each column
    df['source_id'] = df[src].map(labelid)
    df['target_id'] = df[targ].map(labelid)
    return df, labels

def make_sankey(df,threshold, title, *cols):
    # Makes an n-level sankey diagram with n columns
    # Initializing lists to store dataframes and unique labels for each
    # iteration through columns
    df_list = []
    labels = []


    # Iterates through the provided columns and makes a unique dataframe for
    # each one
    for i in range(len(cols)-1):

        src = cols[i]
        targ = cols[i+1]

        # Counts the instances for the specified group of columns
        temp_df = df.groupby([src, targ]).size().reset_index(name='Count')

        # Applies a threshold to remove noise
        temp_df = keep_rows(temp_df, 'Count', '>=', threshold)

        # Utilizes the unique dataframe mapping method
        temp_df, labels = unique_dataframe_mapping(temp_df, src, targ, labels)

        # Appends the dataframe to a list, awaiting concatenation
        df_list.append(temp_df)

    # Concatenates the dataframes for each combination of columns
    df = pd.concat(df_list)

    # Normalize the values to create a color map
    norm = mcolors.Normalize(vmin=df['Count'].min(), vmax=df['Count'].max())
    cmap = cm.get_cmap('viridis')  # Use viridis color map
    link_colors = [mcolors.to_rgba(cmap(norm(val))) for val in df['Count']]

    # Using plotly graph objects to make the sankey
    fig = go.Figure(data=[go.Sankey(
        valueformat=".0f",
        valuesuffix=" Users", # The count of artists is the value
        # Define nodes
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label= labels,
            color=["blue", "green", "purple", "orange"],
        ),
        # Add links
        link=dict(
            source=df.source_id,
            target=df.target_id,
            value= df['Count'],
            color=[f"rgba({int(c[0] * 255)}, "
                   f"{int(c[1] * 255)}, "
                   f"{int(c[2] * 255)},"
                   f" {c[3]})" for c in link_colors]

        ))])
    fig.update_layout(title_text=title,
                  font_size=10)
    return fig

def keep_rows(df, column, operator, comparison):
    # Function for cleaning dataframes, removing specified rows that
    # are related by the operator
        if operator == '==':
            # Keeps rows equal to
            df = df[df[column] == comparison]
        elif operator == '>':
            # Keeps rows greater than
            df = df[df[column] > comparison]
        elif operator == '<':
            # Keeps rows less than
            df = df[df[column] < comparison]
        elif operator == '>=':
            # Keeps rows greater than or equal to
            df = df[df[column] >= comparison]
        elif operator == '<=':
            # Keeps rows less than or equal to
            df = df[df[column] <= comparison]
        else:
            # Keeps rows with comparison
            df = df[df[column] != comparison]
        return df

def bin_loss(df, loss_column, n_bins):
    bins = pd.cut(df[loss_column], bins=n_bins)

    # Bin labels that include loss
    bin_edges = bins.cat.categories
    bin_labels = [f"${int(interval.left):,} - ${int(interval.right):,}"
                  for interval in bin_edges]

    # Assign the bin labels to the bins
    df['loss_bins'] = bins.cat.rename_categories(bin_labels)
    return df

def main():
    print("orange")
    # df = remove_outliers(gambling_stats.merged_df, 'loss')
    # df = bin_loss(df, 'loss', 5)
    # fig = make_sankey(df, 400, "Sankey", 'BirthYear', 'loss_bins')

if __name__ == '__main__':
    main()

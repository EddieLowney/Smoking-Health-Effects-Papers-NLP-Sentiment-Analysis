import plotly.graph_objects as go
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

# fig.update_layout(
#     title_text="Basic Sankey Diagram",
#     font_family="Courier New",
#     font_color="blue",
#     font_size=12,
#     title_font_family="Times New Roman",
#     title_font_color="red",

def _code_mapping(df, src, targ):
    """ Map labels in src and targ columns to integers """
    # Get distinct labels
    labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # Get integer codes
    codes = list(range(len(labels)))

    # Create label to code mapping
    lc_map = dict(zip(labels, codes))

    # Substitute names for codes in dataframe
    df = df.replace({src: lc_map, targ: lc_map})
    return df, labels


def make_sankey(df, src, targ, vals=None, **kwargs):
    """ Generate a sankey diagram
    df - Dataframe
    src - Source column
    targ - Target column
    vals - Values column (optional)
    optional params: pad, thickness, line_color, line_width """

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df[src])  # all 1

    df, labels = _code_mapping(df, src, targ)
    link = {'source': df[src], 'target': df[targ], 'value': values}

    pad = kwargs.get('pad', 50)
    thickness = kwargs.get('thickness', 50)
    line_color = kwargs.get('line_color', 'black')
    line_width = kwargs.get('line_width', 1)

    node = {'label': labels, 'pad': pad, 'thickness': thickness, 'line': {'color': line_color, 'width': line_width}}
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    width = kwargs.get('width', 1600)
    height = kwargs.get('height', 800)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        font_size=25)

    return fig


def show_sankey(df, src, targ, vals=None, **kwargs):
    fig = make_sankey(df, src, targ, vals, **kwargs)
    fig.show()

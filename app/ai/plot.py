import plotly
import plotly.graph_objs as go
import json
import numpy as np


def linear(values):

    x = list(range(len(values)))  # assign x as the dataframe column 'x'
    y = np.array(values)

    data = [
        go.Scatter(x=x, y=y,
                   mode='lines+markers',
                   name='lines+markers')
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

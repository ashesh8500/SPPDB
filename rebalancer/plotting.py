import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

def plot_attractiveness(attractiveness_df: pd.DataFrame):
    sns.barplot(x=attractiveness_df.index, y=attractiveness_df['Attractiveness'], hue=attractiveness_df['Attractiveness'] > 0)
    plt.title('Attractiveness of Stocks')
    plt.show()

def plot_qqq_performance(fig: go.Figure, qqq_data: pd.DataFrame):
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], mode='lines', name='QQQ Performance'))
    return fig

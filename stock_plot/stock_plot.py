#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 株価データ可視化関数 """
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
# from scipy.stats import linregress

import warnings
warnings.filterwarnings("ignore")


def plotly_candlestick(df):
    """
    plotlyでローソク足
    Usage:
        import util
        import pandas as pd
        csv = '7974.JP.csv'

        df = pd.read_csv(csv, index_col='date')  # indexを日付のデータにしないとダメ。型はobject型でもok
        util.plotly_candlestick(df)
    """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df.iloc[:, 0],      # 始値
                                         high=df.iloc[:, 1],      # 高値
                                         low=df.iloc[:, 2],       # 安値
                                         close=df.iloc[:, 3])])   # 終値
    fig.show()


def matplotlib_moving_average_line(df, col_name, short=25, long=75, figsize=(15, 5)):
    """
    matplotlibで移動平均線
    https://medium.com/@yuyasugano/%E4%BB%AE%E6%83%B3%E9%80%9A%E8%B2%A8%E3%81%AE%E8%87%AA%E5%8B%95%E5%8F%96%E5%BC%95%E5%85%A5%E9%96%80-%E3%83%86%E3%82%AF%E3%83%8B%E3%82%AB%E3%83%AB%E6%8C%87%E6%A8%99-53b32f301ce6
    Usage:
        import util
        import pandas as pd
        import matplotlib.pyplot as plt
        csv = '7974.JP.csv'

        df = pd.read_csv(csv, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'Close':'close'})
        fig = util.matplotlib_moving_average_line(df, 'close')  # matplotlibで表示するだけならここまででいい
        plt.show()

        import plotly.offline
        plotly.offline.init_notebook_mode()
        plotly.offline.iplot_mpl(fig)  # matplotlibのPlotly化
        plt.show()
    """
    df['ma_short'] = df[col_name].rolling(window=short).mean()
    df['ma_long'] = df[col_name].rolling(window=long).mean()
    df['diff'] = df['ma_short'] - df['ma_long']
    df['unixtime'] = [datetime.datetime.timestamp(t) for t in df.index]

    # line and Moving Average
    xdate = [x.date() for x in df.index]
    fig = plt.figure(figsize=figsize)
    plt.plot(xdate, df[col_name],label='original')
    plt.plot(xdate, df['ma_long'], label=f'{long}days')
    plt.plot(xdate, df['ma_short'], label=f'{short}days')
    plt.xlim(xdate[0], xdate[-1])
    plt.grid()

    # Cross points
    for i in range(1, len(df)):
        if df.iloc[i - 1]['diff'] < 0 and df.iloc[i]['diff'] > 0:
            print(f'{xdate[i]}:GOLDEN CROSS: {df.iloc[i][col_name]} yen/1stock')
            plt.scatter(xdate[i], df.iloc[i]['ma_short'], marker='o', s=100, color='b')
            plt.scatter(xdate[i], df.iloc[i][col_name], marker='o', s=50, color='b', alpha=0.5)

        if df.iloc[i - 1]['diff'] > 0 and df.iloc[i]['diff'] < 0:
            print(f'{xdate[i]}:DEAD CROSS: {df.iloc[i][col_name]} yen/1stock')
            plt.scatter(xdate[i], df.iloc[i]['ma_short'], marker='o', s=100, color='r')
            plt.scatter(xdate[i], df.iloc[i][col_name], marker='o', s=50, color='r', alpha=0.5)
    plt.legend()
    return fig


def matplotlib_bollinger(df, col_name, window=25, figsize=(15, 5)):
    """
    matplotlibでボリンジャーバンド
    https://medium.com/@yuyasugano/%E4%BB%AE%E6%83%B3%E9%80%9A%E8%B2%A8%E3%81%AE%E8%87%AA%E5%8B%95%E5%8F%96%E5%BC%95%E5%85%A5%E9%96%80-%E3%83%86%E3%82%AF%E3%83%8B%E3%82%AB%E3%83%AB%E6%8C%87%E6%A8%99-53b32f301ce6
    Usage:
        import util
        import pandas as pd
        import matplotlib.pyplot as plt
        csv = '7974.JP.csv'

        df = pd.read_csv(csv, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'Close':'close'})
        fig = util.matplotlib_bollinger(df, 'close')  # matplotlibで表示するだけならここまででいい
        plt.show()

        import plotly.offline
        plotly.offline.init_notebook_mode()
        plotly.offline.iplot_mpl(fig)  # Plotlyではうまく表示できない。。。
        plt.show()
    """
    df['ma'] = df[col_name].rolling(window=window).mean()
    df['sigma'] =  df[col_name].rolling(window=window).std()
    df['ma+2sigma'] = df.ma + 2*df.sigma
    df['ma-2sigma'] = df.ma - 2*df.sigma
    df['diffplus'] = df[col_name] - df['ma+2sigma']
    df['diffminus'] = df['ma-2sigma'] - df[col_name]
    s_up = df[df['diffplus'] > 0][col_name]
    s_down = df[df['diffminus'] > 0][col_name]

    xdate = [x.date() for x in df.index]
    fig = plt.figure(figsize=figsize)
    plt.grid()
    plt.xlim(xdate[0], xdate[-1])
    plt.scatter(s_up.index, s_up.values, marker='x', s=100, color='blue')
    plt.scatter(s_down.index, s_down.values, marker='x', s=100, color='red')
    plt.plot(xdate, df[col_name].values, label='original', color='b', alpha=0.9)
    plt.plot(xdate, df.ma.values, label='{}ma'.format(window))
    plt.fill_between(xdate, df.ma - df.sigma, df.ma + df.sigma, color='red', alpha=0.7, label='$1\sigma$')
    plt.fill_between(xdate, df.ma - 2 * df.sigma, df.ma + 2 * df.sigma, color='red', alpha=0.3, label='$2\sigma$')
    plt.fill_between(xdate, df.ma - 3 * df.sigma, df.ma + 3 * df.sigma, color='red', alpha=0.1, label='$3\sigma$')
    plt.legend()
    return fig

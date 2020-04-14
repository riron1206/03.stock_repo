#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pythonのライブラリpandas_datareaderで指定の株価を取得する
参考:https://www.mazarimono.net/entry/2018/07/20/pandas_datareader

Usage:
    call activate stock
    call python get_stock_days_csv_pandas_datareader.py -i_c code_name.csv
    pause

    # 出力ファイルはマルチカラムなので以下のようにロードする
    # df = pd.read_csv(r'output\out_code_name.csv', index_col=0, header=[0, 1])
    # df["Open", "2424.JP"]
"""
import argparse
import datetime
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web

matplotlib.use('Agg')


def stock_df_plot_plotly(df, out_png='./df.png'):
    import matplotlib.pyplot as plt
    import plotly.offline
    import warnings
    warnings.filterwarnings("ignore")

    plotly.offline.init_notebook_mode()  # matplotlibのPlotly化

    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_axes([0, 0, 1, 1])

    df.plot(ax=axes)  # axes.plot(dfclose)
    for col in df.columns:
        if 'golden_' in col:
            plt.scatter(x=df.index, y=df[col], marker='s', color='red')

    axes.set_xlabel('Time')
    axes.set_ylabel('Price')
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)  # 凡例枠外に書く
    axes.grid()

    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
        print("INFO: save file. [{}]".format(out_png))

    plotly.offline.iplot_mpl(fig)  # matplotlib.pyplotで書いたグラフを、iplot_mpl(fig)と打つだけでPlotlyのインタラクティブなグラフへ変更することができます。
    plt.show()
    plt.clf()
    return


def add_golden_cross_col(df, av_short_col='5MA', av_long_col='30MA'):
    """
    ゴールデンクロス（長期の移動平均線を、短期の移動平均線が下から上に突き抜けたとき）発生時の株価列をデータフレームに追加する
    https://qiita.com/kjybinp1105/items/db4efd07e20000c22f4e
    Args:
        df:pandas_datareaderで取得した株価データフレーム
        av_short_col:dfの短期の移動平均列名
        av_short_col:dfの長期の移動平均列名
    """
    golden_flag_col = f"golden_cross_{av_short_col}_{av_long_col}"
    df[golden_flag_col] = 0
    current_flag = 0
    previous_flag = 1
    for i, price in df.iterrows():
        if(price[av_short_col] > price[av_long_col]):
            current_flag = 1
        else:
            current_flag = 0
        if(current_flag * (1 - previous_flag)):
            df.loc[i, golden_flag_col] = price[av_long_col]
        else:
            df.loc[i, golden_flag_col] = None
        previous_flag = current_flag
    return df


def calc_golden_cross_profit(df, gc_col:str, buy_gc_nday:int, sell_gc_nday:int, out_png=None):
    """
    ゴールデンクロス発生してからn日後に売買したときの利益を計算する
    Args:
        df:pandas_datareaderで取得した株価データフレーム。ゴールデンクロス列が必要
        gc_col:dfのゴールデンクロス列名
        buy_gc_nday:ゴールデンクロス発生日からの日数。 buy_gc_nday 日後に1株だけ購入する
        sell_gc_nday:ゴールデンクロス発生日からの日数。 sell_gc_nday 日後に購入した1株を売却する
        out_png:出力画像パス
    Returns
       株の購入日と売却日、購入日と売却日の終値、1株売買したときの利益の列を持つデータフレーム
    """
    gc_index = df[gc_col].dropna().index.to_list()
    buy_index = list(map(lambda i: i + buy_gc_nday, gc_index))
    sell_index = list(map(lambda i: i + sell_gc_nday, gc_index))

    # 購入日
    df_buy = df.loc[buy_index].reset_index(drop=True)
    df_buy = df_buy[['Date', 'Close']]
    df_buy.columns = ['Date_buy', 'Close_buy']

    # 売却日
    df_sell = df.loc[sell_index].reset_index(drop=True)
    df_sell = df_sell[['Date', 'Close']]
    df_sell.columns = ['Date_sell', 'Close_sell']

    # 利益
    df_profit = pd.concat([df_buy, df_sell], axis=1)
    df_profit['Profit'] = df_profit['Close_sell'] - df_profit['Close_buy']

    if out_png is not None:
        df_profit['Profit'].plot.bar()
        plt.title(f"golden_cross_daylater_buy{buy_gc_nday}_sell{sell_gc_nday}")
        plt.xlabel("old <- golden_cross_count -> recently")
        plt.ylabel('profit(yen) per share of stock')

        plt.savefig(out_png, bbox_inches="tight")
        print("INFO: save file. [{}]".format(out_png))

        plt.show()
        plt.clf()

    print(f"[buy_gc_nday:{buy_gc_nday},sell_gc_nday:{sell_gc_nday}] total profit(yen) per share of stock:", round(np.sum(df_profit['Profit']), 2))
    return df_profit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default='output', help="output dir path.")
    parser.add_argument("-s_y", "--start_year", type=int, default=2000, help="search start year.")
    parser.add_argument("-e_y", "--end_year", type=int, default=None, help="search end year.")
    parser.add_argument("-i_c", "--input_code_csv", type=str, default=None, help="stock brand list csv. e.g. code_name.csv")
    parser.add_argument("-s", "--source", type=str, default='stooq', help="pandas_datareader stock source. e.g. stooq yahoo")
    parser.add_argument("-b", "--brand", type=str, default='6758.JP', help="stock brand id. e.g. 6758.JP")
    args = vars(parser.parse_args())

    os.makedirs(args['output_dir'], exist_ok=True)

    start = datetime.datetime(args['start_year'], 1, 1)
    if args['end_year'] is None:
        end = datetime.datetime.today().strftime("%Y-%m-%d") + ' 00:00:00'
    else:
        end = datetime.datetime(args['end_year'], 1, 1)

    if args['input_code_csv'] is not None:

        dfcode = pd.read_csv(args['input_code_csv'])

        codes = dfcode.iloc[:, 0].tolist()
        if args['source'] == 'stooq':
            # 日本株の場合.JPつける
            codes = [str(c) + '.JP' for c in codes]

        if 'name' in dfcode.columns.tolist():
            names = dfcode['name'].tolist()
        print(codes)
        print(names)

        # pandas_datareaderは1日に250回までしかAPI投げれないらしいので複数銘柄一気にとる
        # https://www.mazarimono.net/entry/2018/10/10/stocks
        df = web.DataReader(codes, args['source'], start, end)
        out_csv = os.path.join(args['output_dir'], 'out_'+args['input_code_csv'])
        df.to_csv(out_csv)
        print("INFO: save file. [{}] {}".format(out_csv, df.shape))

        # 出力ファイルは階層型インデックスで扱いずらいからロードして編集
        df = pd.read_csv(out_csv, skiprows=[2], header=[0, 1])
        df.index = df["Attributes", "Symbols"]
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index)  # インデックスをDatetimeIndexに変換. pandas.DataFrame, pandas.Seriesのインデックスをdatetime64[ns]型にするとDatetimeIndexとみなされ、時系列データを処理する
        df = df.drop(df.columns[[0]], axis=1)
        df = df.reset_index()
        df.set_index('Date', drop=False)  # Date列をindexにし、Date列は残す

        # pandas_datareaderはなぜか期間指定が機能しないので指定する
        df = df[(start <= df['Date']) & (df['Date'] <= end)]

        try:
            # 移動平均線計算.期間内のデータないときエラーになるからtryで囲む
            dfclose = df["Close"]
            for col in dfclose.columns:
                df['5MA', col] = dfclose[col].rolling(window=5, min_periods=0).mean()
            for col in dfclose.columns:
                df['30MA', col] = dfclose[col].rolling(window=30, min_periods=0).mean()
            for col in dfclose.columns:
                df['90MA', col] = dfclose[col].rolling(window=90, min_periods=0).mean()
        except Exception as e:
            print("ERROR:", e)

        df = df.set_index('Date')
        df.to_csv(out_csv)
        print("INFO: save file. [{}] {}".format(out_csv, df.shape))

        # チャート画像化
        stock_df_plot_plotly(df["Close"], out_png=os.path.join(args['output_dir'], 'out_' + pathlib.Path(args['input_code_csv']).stem + '_Close.png'))
        stock_df_plot_plotly(df["5MA"], out_png=os.path.join(args['output_dir'], 'out_' + pathlib.Path(args['input_code_csv']).stem + '_5MA.png'))
        stock_df_plot_plotly(df["30MA"], out_png=os.path.join(args['output_dir'], 'out_' + pathlib.Path(args['input_code_csv']).stem + '_30MA.png'))
        stock_df_plot_plotly(df["90MA"], out_png=os.path.join(args['output_dir'], 'out_' + pathlib.Path(args['input_code_csv']).stem + '_90MA.png'))

    else:
        # FREDからWilshire US REIT指数が取得。一日一本だけの価格
        #df = web.DataReader('WILLREITIND', 'fred', start, end)
        #df.to_csv(os.path.join(args['output_dir'], 'WILLREITIND.csv'), index=False)

        # source=stooqなら日本株データも取れる
        df = web.DataReader(args['brand'], args['source'], start, end)
        df = df.reset_index()
        df.set_index('Date', drop=False)  # Date列をindexにし、Date列は残す
        df = df.sort_values(by=['Date'], ascending=True)  # Dateの昇順にする

        # pandas_datareaderはなぜか期間指定が機能しないので指定する
        df = df[(start <= df['Date']) & (df['Date'] <= end)]
        try:
            # 移動平均線計算.期間内のデータないときエラーになるからtryで囲む
            df['5MA'] = df['Close'].rolling(window=5, min_periods=0).mean()
            df['30MA'] = df['Close'].rolling(window=30, min_periods=0).mean()
            df['90MA'] = df['Close'].rolling(window=90, min_periods=0).mean()
        except Exception as e:
            print("ERROR:", e)

        # ゴールデンクロス列追加
        df = add_golden_cross_col(df, av_short_col='5MA', av_long_col='30MA')
        df = add_golden_cross_col(df, av_short_col='5MA', av_long_col='90MA')
        df = add_golden_cross_col(df, av_short_col='30MA', av_long_col='90MA')

        out_csv = os.path.join(args['output_dir'], args['brand'] + '.csv')
        df.to_csv(out_csv, index=False)
        print("INFO: save file. [{}] {}".format(out_csv, df.shape))

        # チャート画像化
        _df = df[['Date', 'Close', '5MA', '30MA', '90MA', 'golden_cross_5MA_30MA', 'golden_cross_5MA_90MA', 'golden_cross_30MA_90MA']]
        _df = _df.set_index('Date')
        stock_df_plot_plotly(_df, out_png=os.path.join(args['output_dir'], args['brand'] + '.png'))

        # 5日と30日の移動平均線によるゴールデンクロス発生日から1日後に1株購入し、その後9日後に売った時の利益計算
        df = pd.read_csv(out_csv)
        df_profit = calc_golden_cross_profit(df, 'golden_cross_5MA_30MA', 1, 1 + 9,
                                             out_png='golden_cross_profit_buy_gc_nday1_sell_gc_nday10')

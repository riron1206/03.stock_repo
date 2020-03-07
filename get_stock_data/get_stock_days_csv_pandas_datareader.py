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
import os, sys, argparse, datetime, pathlib
import pandas as pd
import pandas_datareader.data as web
import matplotlib
matplotlib.use('Agg')

def stock_df_plot_plotly(df, out_png='./df.png'):
    import matplotlib.pyplot as plt
    import plotly.offline
    import warnings
    warnings.filterwarnings("ignore")

    plotly.offline.init_notebook_mode() # matplotlibのPlotly化
    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_axes([0,0,1,1])
    df.plot(ax=axes)#axes.plot(dfclose)
    axes.set_xlabel('Time')
    axes.set_ylabel('Price')
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)# 凡例枠外に書く
    axes.grid()
    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
        print("INFO: save file. [{}]".format(out_png))
    plotly.offline.iplot_mpl(fig) # matplotlib.pyplotで書いたグラフを、iplot_mpl(fig)と打つだけでPlotlyのインタラクティブなグラフへ変更することができます。
    plt.show()
    plt.clf()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default='output', help="output dir path.")
    parser.add_argument("-s_y", "--start_year", type=int, default=2000, help="search start year.")
    parser.add_argument("-e_y", "--end_year", type=int, default=None, help="search end year.")
    parser.add_argument("-i_c", "--input_code_csv", type=str, default=None, help="stock brand list csv. e.g. code_name.csv")
    parser.add_argument("-s", "--source", type=str, default='stooq', help="stock source. e.g. stooq")
    parser.add_argument("-b", "--brand", type=str, default='6758.JP', help="stock brand id. e.g. 6758.JP")
    args = vars(parser.parse_args())

    os.makedirs(args['output_dir'], exist_ok=True)

    start = datetime.datetime(args['start_year'], 1, 1)
    if args['end_year'] is None:
        end = datetime.datetime.today().strftime("%Y-%m-%d")+' 00:00:00'
    else:
        end = datetime.datetime(args['end_year'], 1, 1)

    if args['input_code_csv'] is not None:

        dfcode = pd.read_csv(args['input_code_csv'])

        codes = dfcode.iloc[:,0].tolist()
        if args['source'] == 'stooq':
            # 日本株の場合.JPつける
            codes = [str(c)+'.JP' for c in codes]

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
        df.index = pd.to_datetime(df.index) # インデックスをDatetimeIndexに変換. pandas.DataFrame, pandas.Seriesのインデックスをdatetime64[ns]型にするとDatetimeIndexとみなされ、時系列データを処理する
        df = df.drop(df.columns[[0]], axis=1)
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
        df.to_csv(out_csv)
        print("INFO: save file. [{}] {}".format(out_csv, df.shape))
        # チャート画像化
        stock_df_plot_plotly(df["Close"], out_png=os.path.join(args['output_dir'], 'out_'+pathlib.Path(args['input_code_csv']).stem+'_Close.png'))
        stock_df_plot_plotly(df["5MA"], out_png=os.path.join(args['output_dir'], 'out_'+pathlib.Path(args['input_code_csv']).stem+'_5MA.png'))
        stock_df_plot_plotly(df["30MA"], out_png=os.path.join(args['output_dir'], 'out_'+pathlib.Path(args['input_code_csv']).stem+'_30MA.png'))
        stock_df_plot_plotly(df["90MA"], out_png=os.path.join(args['output_dir'], 'out_'+pathlib.Path(args['input_code_csv']).stem+'_90MA.png'))

    else:
        # FREDからWilshire US REIT指数が取得。一日一本だけの価格
        #df = web.DataReader('WILLREITIND', 'fred', start, end)
        #df.to_csv(os.path.join(args['output_dir'], 'WILLREITIND.csv'), index=False)

        # source=stooqなら日本株データも取れる
        df = web.DataReader(args['brand'], args['source'], start, end)
        try:
            # 移動平均線計算.期間内のデータないときエラーになるからtryで囲む
            df['5MA'] = df['Close'].rolling(window=5, min_periods=0).mean()
            df['30MA'] = df['Close'].rolling(window=30, min_periods=0).mean()
            df['90MA'] = df['Close'].rolling(window=90, min_periods=0).mean()
        except Exception as e:
            print("ERROR:", e)
        out_csv = os.path.join(args['output_dir'], args['brand']+'.csv')
        df.to_csv(out_csv)
        print("INFO: save file. [{}] {}".format(out_csv, df.shape))
        # チャート画像化
        _df = df[['Close','5MA','30MA','90MA']]
        stock_df_plot_plotly(_df, out_png=os.path.join(args['output_dir'], args['brand']+'.png'))

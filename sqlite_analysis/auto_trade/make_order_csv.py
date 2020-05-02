#!/usr/bin/env python
# coding: utf-8
"""
自動売買で使うcsvを作成
Usage:
    $ activate stock
    $ python ./make_order_csv.py -all  # 最近のデータだけで全銘柄実行
"""
import argparse
import datetime
import glob
import os
import sqlite3
import pathlib
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class SqlliteMgr():
    def __init__(self, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
        self.db_file_name = db_file_name

    def table_to_df(self, table_name=None, sql=None):
        """ sqlite3で指定テーブルのデータをDataFrameで返す """
        conn = sqlite3.connect(self.db_file_name)
        if table_name is not None:
            return pd.read_sql(f'SELECT * FROM {table_name}', conn)
        elif sql is not None:
            return pd.read_sql(sql, conn)
        else:
            return None

    def get_code_price(self, code, start_date, end_date):
        """ DBから指定銘柄の株価取得 """
        sql = f"""
        SELECT
            *
        FROM
            prices AS t
        WHERE
            t.code = {code}
        AND
            t.date BETWEEN '{start_date}' AND '{end_date}'
        """
        # print(sql)
        return self.table_to_df(sql=sql)


def buy_pattern2(row):
    """
    買い条件:
    - 当日終値が5MA,25MAより上
    - 当日終値が前日からの直近10日間の最大高値より上
    - 当日出来高が前日比30%以上増
    - 当日終値が当日の値幅の上位70%以上(大陽線?)
    - 当日終値が25MA乖離率+5％未満
    """
    if row['close'] >= row['5MA'] \
        and row['close'] >= row['25MA'] \
        and row['close'] >= row['high_shfi1_10MAX'] \
        and row['volume_1diff_rate'] >= 0.3 \
        and row['close'] >= ((row['high'] - row['low']) * 0.7) + row['low'] \
        and (row['close'] - row['25MA']) / row['25MA'] < 0.05:
        return row
    else:
        return pd.Series()


def buy_pattern2_1(row, x=10):
    """
    買い条件:
    - buy_pattern2()の条件
    - 翌日始値が前日終値+x円で指値注文
    """
    row = buy_pattern2(row)
    if row.empty == False:
        row['buy_limit_price'] = row['close'] + x
        return row
    else:
        return pd.Series()


def sell_pattern2(df, df_buy):
    """
    売り条件:
    - 購入シグナルの出た前日or当日の安値の-5円を下回ったら損切
    - 購入シグナル当日の25MAの+5～+8%を超えたら利確
    - 購入シグナル当日終値の上にある75MAと200MAに最新データがタッチしたら利確
    - 5MAを割込んだら損切
    Args:
        df: 1銘柄の株価データ
        df_buy: dfの購入シグナルが出た行だけを取り出したデータフレーム
    Returns:
        注文するIFDOCOの利確・損切値の値を詰めたデータフレーム
    """
    ifdoco_set_profits, ifdoco_stop_losses = [], []

    for ind in df_buy.index:
        # 購入シグナル当日の25MAとの乖離(+5～+8%)を超えた値をIFDOCO-成行の利確とする(利確)
        ifdoco_set_profit = df.loc[ind]['25MA'] * 1.08  # 8%にしておく

        # 購入シグナルの出た前日or当日の安値の-5円をIFDOCO-成行の損切とする（損切）
        stop_loss2 = df.loc[ind]['low'] - 5
        if ind != 0:
            stop_loss1 = df.loc[ind - 1]['low'] - 5
        else:
            stop_loss1 = stop_loss2
        if stop_loss1 < stop_loss2:
            ifdoco_stop_loss = stop_loss2
        else:
            ifdoco_stop_loss = stop_loss1

        # シグナル当日終値の上にある75MA、200MAの大きい方を通常-指値とする(利確)
        if (df.loc[ind]['close'] <= df.iloc[ind]['75MA']) \
            or (df.loc[ind]['close'] <= df.iloc[ind]['200MA']):

            if df.iloc[ind]['75MA'] > df.iloc[ind]['200MA']:
                limit_price = df.iloc[ind]['75MA']
            else:
                limit_price = df.iloc[ind]['200MA']

            if ifdoco_set_profit < limit_price:
                ifdoco_set_profit = limit_price

        # シグナル当日の終値の下にある5MAを通常-逆指値とする（損切）
        if df.iloc[ind]['5MA'] < df.iloc[ind]['close']:
            stop_loss = df.iloc[ind]['5MA']

            if ifdoco_stop_loss < stop_loss:
                ifdoco_stop_loss = stop_loss

        ifdoco_set_profits.append(ifdoco_set_profit)
        ifdoco_stop_losses.append(ifdoco_stop_loss)

    df_buy['ifdoco_set_profit'] = ifdoco_set_profits
    df_buy['ifdoco_stop_loss'] = ifdoco_stop_losses

    return df_buy


def set_buy_df_for_auto2(df_buy, minimum_buy_threshold, under_unit, kubun='現物'):
    """ 自動売買用のcsv列作成 買い用 """
    df_buy = df_buy.reset_index(drop=True).reset_index()
    df_buy = df_buy.rename(columns={'index': '注文番号', 'code': '証券コード'})
    df_buy['注文番号'] = df_buy['注文番号'] + 1
    df_buy['シグナル日'] = df_buy['date']
    df_buy['取引方法'] = '新規'
    df_buy['取引区分'] = kubun + '買い'
    df_buy['信用取引区分'] = ''
    df_buy['注文条件１'] = 'IFDOCO'
    df_buy['注文条件２'] = '成行'

    # 注文数指定  最低でもunder_unit枚買う 切り捨て
    df_buy['注文数'] = [(minimum_buy_threshold // (v * under_unit)) * under_unit for v in df_buy['close']]
    df_buy['注文数'] = df_buy['注文数'].astype(int)

    if df_buy['buy_limit_price'].all() != 0:
        df_buy['注文条件２'] = '指値'
        df_buy['指値'] = df_buy['buy_limit_price']
    else:
        df_buy['指値'] = ''

    df_buy['利確'] = df_buy['ifdoco_set_profit']
    df_buy['損切'] = df_buy['ifdoco_stop_loss']
    df_buy['注文日'] = ''
    df_buy['約定日'] = ''
    df_buy = df_buy.sort_values(by=['シグナル日'])
    return df_buy


def calc_code_order(code, start_date, end_date, pattern, minimum_buy_threshold, under_unit):
    """ 1銘柄について、注文日と利確損切ラインを見つける """
    # DBから株価取得
    sqlmgr = SqlliteMgr()
    df = sqlmgr.get_code_price(code, start_date, end_date)

    # 買い条件列追加
    df['5MA'] = df['close'].rolling(window=5).mean()
    df['25MA'] = df['close'].rolling(window=25).mean()
    df['75MA'] = df['close'].rolling(window=75).mean()
    df['200MA'] = df['close'].rolling(window=200).mean()
    df['close_10MAX'] = df['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
    df['high_10MAX'] = df['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
    df['high_shfi1_10MAX'] = df['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
    df['high_shfi1_15MAX'] = df['high'].shift(1).fillna(0).rolling(window=15, min_periods=0).max()  # 前日からの直近15日間の中で最大高値
    df['volume_1diff_rate'] = (df['volume'] - df['volume'].shift(1).fillna(0)) / df['volume']  # 前日比出来高
    df['open_close_1diff'] = df['open'].shift(-1).fillna(0) - df['close']  # 翌日始値-当日終値
    df['high_shift_1'] = df['high'].shift(-1).fillna(0)  # 翌日高値
    df['buy_limit_price'] = 0  # 買いの指値の値

    # pattern関数の条件にマッチしたレコードだけ抽出（購入）
    if pattern == 2:
        df_buy = df.apply(buy_pattern2, axis=1).dropna(how='any')
        if df_buy.empty == False:
            df_buy = sell_pattern2(df, df_buy)
            df_buy = set_buy_df_for_auto2(df_buy, minimum_buy_threshold, under_unit)
        else:
            df_buy = None
    elif pattern == 2_1:
        df_buy = df.apply(buy_pattern2_1, axis=1).dropna(how='any')
        if df_buy.empty == False:
            df_buy = sell_pattern2(df, df_buy)
            df_buy = set_buy_df_for_auto2(df_buy, minimum_buy_threshold, under_unit)
        else:
            df_buy = None

    return df_buy


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default='input')
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[2914])  # JT
    ap.add_argument("-sd", "--start_date", type=str, default=None)  # '2015-01-01'
    ap.add_argument("-ed", "--end_date", type=str, default=None)
    ap.add_argument("-p", "--pattern", type=int, default=2_1)
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000)
    ap.add_argument("-uu", "--under_unit", type=int, default=100)
    ap.add_argument("-k", "--kubun", type=str, default='現物')
    ap.add_argument("-all", "--is_all_codes", action='store_const', const=True, default=False, help="all stock flag.")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    # code = 7267  # リクルート
    # code = 6981  # 村田
    # code = 2914  # JT
    # code = 6758  # Sony
    codes = args['codes']
    if args['is_all_codes']:
        codes = [int(pathlib.Path(p).stem) for p in glob.glob(r'D:\DB_Browser_for_SQLite\csvs\kabuoji3\*.csv')]

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    end_date = args['end_date']
    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")  # 今日の日付

    start_date = args['start_date']
    if start_date is None:
        start_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date() - datetime.timedelta(days=500)  # 500日前からスタート
        start_date = start_date.strftime("%Y-%m-%d")
    print('start_date, end_date', start_date, end_date)

    df_buys = None
    pbar = tqdm(codes)
    for code in pbar:
        pbar.set_description(str(code))
        df_buy = calc_code_order(code, start_date, end_date, args['pattern'], args['minimum_buy_threshold'], args['under_unit'])
        if df_buy is not None:
            if df_buys is None:
                df_buys = df_buy
            else:
                df_buys = pd.concat([df_buys, df_buy], ignore_index=True)

    if df_buys is not None:
        df_buys['minimum_buy_threshold'] = args['minimum_buy_threshold']  # 予算も列に入れとく

        df_buys = df_buys.dropna(subset=['シグナル日'])
        df_buys = df_buys.sort_values(by=['取引区分'], ascending=False)
        df_buys = df_buys.sort_values(by=['シグナル日', '証券コード'])
        df_buys['注文番号'] = range(1, df_buys.shape[0] + 1)
        df_buys = df_buys[['シグナル日', '注文番号', '証券コード', '取引方法', '取引区分', '信用取引区分',
                           '注文数', '注文条件１', '注文条件２', '指値', '利確', '損切', '注文日', '約定日',
                           'close', 'volume', 'minimum_buy_threshold']]

        df_buys['シグナル日'] = pd.to_datetime(df_buys['シグナル日'])
        run_day = df_buys['シグナル日'].iloc[-1]  # 最後の日だけ残す
        df_buys = df_buys[df_buys['シグナル日'] >= run_day]

        df_buys = df_buys.sort_values(by=['volume'], ascending=False)  # 出来高でかい順にしておく
        df_buys.to_csv(os.path.join(output_dir, 'auto_order.csv'), index=False, encoding='shift-jis')
#!/usr/bin/env python
# coding: utf-8
"""
1銘柄について、指定の株価抽出条件にマッチした株価からn日後に売買したときの損益を計算する
Usage:
    $ activate stock
    $ python ./simple_trade.py -cs 6758 -p 2
    $ python ./simple_trade.py -cs 6758 7974 -p 2
"""
import argparse
import sqlite3
import pandas as pd


def table_to_df(table_name=None, sql=None, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """ sqlite3で指定テーブルのデータをDataFrameで返す """
    conn = sqlite3.connect(db_file_name)
    if table_name is not None:
        return pd.read_sql(f'SELECT * FROM {table_name}', conn)
    elif sql is not None:
        return pd.read_sql(sql, conn)
    else:
        return None


def pattern1(row):
    """
    株価抽出条件:
    - 当日終値が5MA,25MAより上
    - 当日終値が直近10日間の中で最大
    - 当日出来高が前日比30%以上増
    - 当日終値が当日の値幅の上位70%以上(大陽線?)
    - 当日終値が25MA乖離率+5％未満
    - 翌日始値-当日終値が0より大きい
    """
    if row['close'] >= row['5MA'] \
        and row['close'] >= row['25MA'] \
        and row['close'] == row['close_10MAX'] \
        and row['volume_1diff_rate'] >= 0.3 \
        and row['close'] >= ((row['high'] - row['low']) * 0.7) + row['low'] \
        and (row['close'] - row['25MA']) / row['25MA'] < 0.05 \
        and row['open_close_1diff'] > 0:
        return row
    else:
        return pd.Series()


def pattern2(row):
    """
    株価抽出条件:
    - 当日終値が5MA,25MAより上
    - 当日終値が前日からの直近10日間の最大高値より上
    - 当日出来高が前日比30%以上増
    - 当日終値が当日の値幅の上位70%以上(大陽線?)
    - 当日終値が25MA乖離率+5％未満
    - 翌日始値-当日終値が0より大きい
    """
    if row['close'] >= row['5MA'] \
        and row['close'] >= row['25MA'] \
        and row['close'] >= row['high_shfi1_10MAX'] \
        and row['volume_1diff_rate'] >= 0.3 \
        and row['close'] >= ((row['high'] - row['low']) * 0.7) + row['low'] \
        and (row['close'] - row['25MA']) / row['25MA'] < 0.05 \
        and row['open_close_1diff'] > 0:
        return row
    else:
        return pd.Series()


def pattern3(row):
    """
    株価抽出条件:
    - 当日終値が5MA,25MAより上
    - 当日終値が前日からの直近15日間の最大高値より上
    - 当日出来高が前日比30%以上増
    - 当日終値が当日の値幅の上位70%以上(大陽線?)
    - 当日終値が25MA乖離率+5％未満
    - 翌日始値-当日終値が0より大きい
    """
    if row['close'] >= row['5MA'] \
        and row['close'] >= row['25MA'] \
        and row['close'] >= row['high_shfi1_15MAX'] \
        and row['volume_1diff_rate'] >= 0.3 \
        and row['close'] >= ((row['high'] - row['low']) * 0.7) + row['low'] \
        and (row['close'] - row['25MA']) / row['25MA'] < 0.05 \
        and row['open_close_1diff'] > 0:
        return row
    else:
        return pd.Series()


def calc_buy_sell_df(base_index, df, buy_nday=1, sell_nday=5):
    """
    売買時のレコードを抜き出し、引き算して利益を計算する
    利益を計算したデータフレームを返す
    """
    buy_index = list(map(lambda i: i + buy_nday, base_index))  # base_indexのbuy_nday日後のレコード
    sell_index = list(map(lambda i: i + sell_nday, base_index))  # base_indexのsell_nday日後のレコード

    df_buy = df.loc[buy_index].reset_index(drop=True)
    df_buy = df_buy[['date', 'open']]
    df_buy.columns = ["購入日", "購入金額"]

    df_sell = df.loc[sell_index].reset_index(drop=True)
    df_sell = df_sell[['date', 'close']]
    df_sell.columns = ["売却日", "売却金額"]

    df_profit = pd.concat([df_buy, df_sell], axis=1)
    df_profit['損益'] = df_profit['売却金額'] - df_profit['購入金額']
    return df_profit


def main(code, start_date, pattern):
    # DBから株価取得
    sql = f"""
    SELECT
        *
    FROM
        prices AS t
    WHERE
        t.code = {code}
    AND
        t.date > '{start_date}'
    """
    df = table_to_df(sql=sql)

    # 売買条件列追加
    df['5MA'] = df['close'].rolling(window=5).mean()
    df['25MA'] = df['close'].rolling(window=25).mean()
    df['close_10MAX'] = df['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
    df['high_10MAX'] = df['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
    df['high_shfi1_10MAX'] = df['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
    df['high_shfi1_15MAX'] = df['high'].shift(1).fillna(0).rolling(window=15, min_periods=0).max()  # 前日からの直近15日間の中で最大高値
    df['volume_1diff_rate'] = (df['volume'] - df['volume'].shift(1).fillna(0)) / df['volume']  # 前日比出来高
    df['open_close_1diff'] = df['open'].shift(-1).fillna(0) - df['close']  # 翌日始値-当日終値

    # pattern関数の条件にマッチしたレコードだけ抽出
    if pattern == 1:
        df_pattern = df.apply(pattern1, axis=1).dropna(how='any')
    elif pattern == 2:
        df_pattern = df.apply(pattern2, axis=1).dropna(how='any')
    elif pattern == 3:
        df_pattern = df.apply(pattern3, axis=1).dropna(how='any')

    # 損益計算
    # 条件に当てはまるレコード無ければ、行もしくは列がないので損益計算しない
    if df_pattern.shape[0] != 0 and df_pattern.shape[1] != 0:
        df_pattern = df_pattern[df.columns]
        pattern = df_pattern.dropna().index.to_list()
        # print('df_pattern index:', pattern)

        df_profit = calc_buy_sell_df(pattern, df)
        print(df_profit)
        print(f"code: {code} 損益合計: {round(df_profit['損益'].sum())}\n")
    else:
        print(f'\ncode: {code} 売買条件に当てはまるレコードなし\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[7267])
    ap.add_argument("-s", "--start_date", type=str, default='2017-12-31')
    ap.add_argument("-p", "--pattern", type=int, default=1)
    args = vars(ap.parse_args())

    # code = 7267  # リクルート
    # code = 6981  # 村田
    # code = 2914  # JT
    # code = 6758  # Sony
    codes = args['codes']

    # start_date = '2017-12-31'
    start_date = args['start_date']

    pattern = args['pattern']

    for code in codes:
        main(code, start_date, pattern)

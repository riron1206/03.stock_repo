#!/usr/bin/env python
# coding: utf-8
"""
1銘柄について、指定の株価抽出条件にマッチした株価からn日後に売買したときの損益を計算する
Usage:
    $ activate stock
    $ python ./simple_trade.py -cs 6758 -bp 2
    $ python ./simple_trade.py -cs 7013 8304 3407 2502 2802 4503 6857 6113 6770 8267 7202 5019 8001 4208 4523 5201 9202 6472 9613 9437 6361 8725 2413 6103 9532 3861 4578 1802 6703 9007 6645 7733 4452 6952 1812 9107 7012 9503 2801 7751 6971 4151 2503 6326 3405 8253 9008 9009 9433 5406 1605 9766 4902 6301 1721 7186 4751 2501 3436 6674 5411 5020 6473 3086 4507 8355 4911 7762 1803 9104 4004 4063 8303 5401 9412 7735 7269 7270 5232 4005 5713 6302 8053 5802 8830 6724 1928 9735 4689 3382 2768 6758 8729 9984 8630 4568 8750 6367 1801 7912 4506 5541 5233 6976 1925 8601 8233 2531 4502 8331 4519 9502 8795 2432 3401 6762 4631 4543 4061 6902 4324 5301 9022 3289 8035 8766 9531 9005 8804 9501 4042 5332 9001 9602 5707 5901 3101 3402 5714 4043 7911 7203 8015 4704 7731 9021 2871 1963 4021 7201 2002 3105 6988 5202 5333 4272 5703 1332 6471 3863 5631 4041 2914 9062 6701 5214 9432 2282 6178 9101 8604 1808 6752 7832 9020 6305 6501 7004 7205 9983 6954 8028 8354 5803 6702 6504 4901 5108 5801 7267 8628 7261 8252 1333 8002 8411 7003 4183 5706 8309 8316 8031 8801 3099 4188 7211 7011 8058 9301 8802 6503 5711 8306 6479 2269 6506 9064 7951 7272 3103 6841 5101 6098 7752 8308
"""
import argparse
import sqlite3
import pandas as pd
from tqdm import tqdm


def table_to_df(table_name=None, sql=None, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """ sqlite3で指定テーブルのデータをDataFrameで返す """
    conn = sqlite3.connect(db_file_name)
    if table_name is not None:
        return pd.read_sql(f'SELECT * FROM {table_name}', conn)
    elif sql is not None:
        return pd.read_sql(sql, conn)
    else:
        return None


def buy_pattern1(row):
    """
    買い条件:
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


def buy_pattern3(row):
    """
    買い条件:
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

    df_profit = pd.concat([df_buy, df_sell], axis=1).dropna(how='any')
    df_profit['損益'] = df_profit['売却金額'] - df_profit['購入金額']
    return df_profit


def trade(code, start_date, pattern):
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
    df['75MA'] = df['close'].rolling(window=75).mean()
    df['200MA'] = df['close'].rolling(window=200).mean()
    df['close_10MAX'] = df['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
    df['high_10MAX'] = df['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
    df['high_shfi1_10MAX'] = df['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
    df['high_shfi1_15MAX'] = df['high'].shift(1).fillna(0).rolling(window=15, min_periods=0).max()  # 前日からの直近15日間の中で最大高値
    df['volume_1diff_rate'] = (df['volume'] - df['volume'].shift(1).fillna(0)) / df['volume']  # 前日比出来高
    df['open_close_1diff'] = df['open'].shift(-1).fillna(0) - df['close']  # 翌日始値-当日終値

    # pattern関数の条件にマッチしたレコードだけ抽出
    if pattern == 1:
        df_pattern = df.apply(buy_pattern1, axis=1).dropna(how='any')
    elif pattern == 2:
        df_pattern = df.apply(buy_pattern2, axis=1).dropna(how='any')
    elif pattern == 3:
        df_pattern = df.apply(buy_pattern3, axis=1).dropna(how='any')

    # 損益計算
    # 条件に当てはまるレコード無ければ、行もしくは列がないので損益計算しない
    if df_pattern.shape[0] != 0 and df_pattern.shape[1] != 0:
        df_pattern = df_pattern[df.columns]
        pattern = df_pattern.dropna().index.to_list()
        # print('df_pattern index:', pattern)

        df_profit = calc_buy_sell_df(pattern, df)
        print(df_profit)
        print(f"code: {code} 損益合計: {round(df_profit['損益'].sum())}\n")

        df_profit['code'] = code
        return df_profit
    else:
        print(f'\ncode: {code} 売買条件に当てはまるレコードなし\n')
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[7267])
    ap.add_argument("-s", "--start_date", type=str, default='2015-01-01')
    ap.add_argument("-bp", "--buy_pattern", type=int, default=2)
    args = vars(ap.parse_args())

    # code = 7267  # リクルート
    # code = 6981  # 村田
    # code = 2914  # JT
    # code = 6758  # Sony
    codes = args['codes']

    # start_date = '2017-12-31'
    start_date = args['start_date']

    pattern = args['buy_pattern']

    df_profit_codes = None
    for code in tqdm(codes):
        df_profit = trade(code, start_date, pattern)

        if df_profit_codes is None:
            if df_profit is not None:
                df_profit_codes = df_profit
        else:
            if df_profit is not None:
                df_profit_codes = pd.concat([df_profit_codes, df_profit], ignore_index=True)
    df_profit_codes = df_profit_codes[['code', '購入日', '購入金額', '売却日', '売却金額', '損益']]
    df_profit_codes.to_csv(r'output\simple_trade.csv', index=False, encoding='shift-jis')
    print('損益合計:', df_profit_codes['損益'].sum())

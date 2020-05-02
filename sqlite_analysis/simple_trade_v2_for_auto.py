#!/usr/bin/env python
# coding: utf-8
"""
複数銘柄について、指定の買い売り条件にマッチした株価を売買したときの損益を計算する
自動売買で使うcsvも出す
→buy_pattern2() とsell_pattern1() のやり方は持ち金制限なしなら儲かるが、1日の購入金額100万以内にするとマイナスになる
Usage:
    $ activate stock
    $ python ./simple_trade_v2_for_auto.py -cs 6091 -bp 2
    $ python ./simple_trade_v2_for_auto.py -cs 6091 -bp 2_1 -sp 1_1
    $ python ./simple_trade_v2_for_auto.py -cs 4507 -sd 20141001 -ed 20151112
    $ python ./simple_trade_v2_for_auto.py -cs 7013 -bp 2
    $ python ./simple_trade_v2_for_auto.py -sd 20160101 -cs 4151
    $ python ./simple_trade_v2_for_auto.py -cs 2914 -sd 20150101
    $ python ./simple_trade_v2_for_auto.py -cs 2914 -sd 20190101 -dep 0
    $ python ./simple_trade_v2_for_auto.py -cs 2914 7013 8304 -bp 2 -sd 20141001
    $ python ./simple_trade_v2_for_auto.py -dep 0 -cs 7013 8304 3407 2502 2802 4503 6857 6113 6770 8267 7202 5019 8001 4208 4523 5201 9202 6472 9613 9437 6361 8725 2413 6103 9532 3861 4578 1802 6703 9007 6645 7733 4452 6952 1812 9107 7012 9503 2801 7751 6971 4151 2503 6326 3405 8253 9008 9009 9433 5406 1605 9766 4902 6301 1721 7186 4751 2501 3436 6674 5411 5020 6473 3086 4507 8355 4911 7762 1803 9104 4004 4063 8303 5401 9412 7735 7269 7270 5232 4005 5713 6302 8053 5802 8830 6724 1928 9735 4689 3382 2768 6758 8729 9984 8630 4568 8750 6367 1801 7912 4506 5541 5233 6976 1925 8601 8233 2531 4502 8331 4519 9502 8795 2432 3401 6762 4631 4543 4061 6902 4324 5301 9022 3289 8035 8766 9531 9005 8804 9501 4042 5332 9001 9602 5707 5901 3101 3402 5714 4043 7911 7203 8015 4704 7731 9021 2871 1963 4021 7201 2002 3105 6988 5202 5333 4272 5703 1332 6471 3863 5631 4041 2914 9062 6701 5214 9432 2282 6178 9101 8604 1808 6752 7832 9020 6305 6501 7004 7205 9983 6954 8028 8354 5803 6702 6504 4901 5108 5801 7267 8628 7261 8252 1333 8002 8411 7003 4183 5706 8309 8316 8031 8801 3099 4188 7211 7011 8058 9301 8802 6503 5711 8306 6479 2269 6506 9064 7951 7272 3103 6841 5101 6098 7752 8308
    $ python ./simple_trade_v2_for_auto.py -dep 0 -all  # 最近のデータだけで全銘柄実行
"""
import argparse
import datetime
import glob
import os
import sqlite3
import pathlib
import numpy as np
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


def edit_df_buy_pattern2(df_buy, df_sell):
    """ buy_pattern2()のdf_buyに注文条件１, 注文条件２, 利確, 損切を入れる """

    df_buy = pd.merge(df_buy, df_sell, on=['code', 'date', '注文数'])

    for _ind, _row in df_buy.fillna(0).iterrows():
        # 通常の逆指値、通常の指値、IFDOCOの損切ライン、
        stop_loss = _row['stop_loss']
        limit_price = _row['limit_price']
        ifdoco_stop_loss = _row['ifdoco_stop_loss']
        ifdoco_set_profit = _row['ifdoco_set_profit']
        # print(stop_loss, limit_price, ifdoco_stop_loss, ifdoco_set_profit)

        df_buy.at[_ind, '損切'] = ifdoco_stop_loss
        if stop_loss > ifdoco_stop_loss:
            df_buy.at[_ind, '損切'] = stop_loss

        df_buy.at[_ind, '利確'] = ifdoco_set_profit
        if 0 < limit_price < ifdoco_set_profit:
            df_buy.at[_ind, '利確'] = limit_price

        df_buy.at[_ind, '注文条件１'] = 'IFDOCO'
        df_buy.at[_ind, '注文条件２'] = '成行'

    return df_buy


def buy_pattern2_1(row, x=20):
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


def sell_pattern_calc_profit_loss(df, df_sell):
    """
    sell_pattern()の損益計算
    Args:
        df: 1銘柄の株価データ
        df_sell: sell_pattern()のデータフレーム
        　　　　　※注文するIDOCOの利確・損切値、通常指値、通常逆指値の値の列を持つ形じゃないと動かない
    """

    buy_dates, buy_values, units = [], [], []
    sell_orders, sell_dates, sell_values, sell_condition1s, sell_condition2s = [], [], [], [], []

    for signal_ind in df_sell.index:

        # シグナルがちょうど最終レコードだと、次の日購入の計算はできないから None入れとく
        if signal_ind == df.shape[0] - 1:
            _today = df.loc[signal_ind]['date']
            _tomorrow = datetime.datetime.strptime(_today, '%Y-%m-%d').date() + datetime.timedelta(days=1)
            _tomorrow = _tomorrow.strftime("%Y-%m-%d")
            buy_dates.append(_tomorrow)
            sell_orders.append(_tomorrow)
            units.append(df_sell.loc[signal_ind]['注文数'])
            buy_values.append(None)
            sell_dates.append(None)
            sell_values.append(None)
            sell_condition1s.append(None)
            sell_condition2s.append(None)
            continue

        # 通常の逆指値、通常の指値、IFDOCOの損切ライン、
        stop_loss = df_sell.loc[signal_ind]['stop_loss']
        limit_price = df_sell.loc[signal_ind]['limit_price']
        ifdoco_stop_loss = df_sell.loc[signal_ind]['ifdoco_stop_loss']
        ifdoco_set_profit = df_sell.loc[signal_ind]['ifdoco_set_profit']

        # 買いが指値か成行か. buy_limit_price列あれば指値で買う
        buy_ind = None
        if 'buy_limit_price' in df_sell.columns:
            # 指値
            # シグナル翌日から株価のデータフレームループ
            for _ind, _row in df[signal_ind + 1:].iterrows():

                # 翌営業日の始値が指値未満なら購入できない
                if _row['open'] < df_sell.loc[signal_ind]['buy_limit_price']:
                    continue

                buy_dates.append(_row['date'])
                sell_orders.append(_row['date'])
                buy_values.append(_row['open'])
                buy_ind = _ind
                break
        else:
            # 成行
            # 翌営業日に購入
            buy_dates.append(df.loc[signal_ind + 1]['date'])
            # 注文日は購入日と同じにする
            sell_orders.append(df.loc[signal_ind + 1]['date'])
            # 購入金額はシグナルの次の日の始値
            buy_values.append(df.loc[signal_ind + 1]['open'])
            # 購入日のインデックス
            buy_ind = signal_ind + 1

        if buy_ind is None:
            continue

        # 注文数
        units.append(df_sell.loc[signal_ind]['注文数'])

        # 購入日から株価のデータフレームループ
        for _ind, _row in df[buy_ind:].iterrows():

            # ------------ 通常逆指値 通常の逆指値を下回るか ------------ #
            if _row['close'] <= stop_loss:
                sell_condition1 = '通常'
                sell_condition2 = '逆指値'  # 損切
                sell_date = _row['date']
                sell_value = stop_loss
                # 始値も stop_loss 下回った場合
                if _row['open'] <= stop_loss:
                    sell_value = _row['open']

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition1s.append(sell_condition1)
                sell_condition2s.append(sell_condition2)
                break

            # ------------ 通常指値 通常の指値を上回るか ------------ #
            if _row['high'] >= limit_price:
                sell_condition1 = '通常'
                sell_condition2 = '指値'
                sell_date = _row['date']
                sell_value = limit_price
                # 始値も limit_price 上回った場合
                if _row['open'] >= limit_price:
                    sell_value = _row['open']

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition1s.append(sell_condition1)
                sell_condition2s.append(sell_condition2)
                break

            # ------------ IFDOCOの損切ラインを下回るか ------------ #
            if _row['low'] <= ifdoco_stop_loss:
                sell_condition1 = 'IFDOCO'
                sell_condition2 = '成行'
                sell_date = None
                sell_value = None

                # 当日以降に売却
                for _i, _r in df[_ind:].iterrows():
                    sell_date = df.loc[_i]['date']
                    sell_value = None
                    # その日中に stop_loss 下回った場合
                    if df.loc[_i]['low'] <= ifdoco_stop_loss and df.loc[_i]['open'] >= ifdoco_stop_loss:
                        sell_value = ifdoco_stop_loss
                        break
                    # 始値が stop_loss 下回った場合
                    if df.loc[_i]['open'] < ifdoco_stop_loss:
                        sell_value = df.loc[_i]['open']
                        break

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition1s.append(sell_condition1)
                sell_condition2s.append(sell_condition2)
                break

            # ------------ IFDOCOの利確ライン上回るか ------------ #
            if _row['high'] >= ifdoco_set_profit:
                sell_condition1 = 'IFDOCO'
                sell_condition2 = '成行'
                sell_date = None
                sell_value = None

                # 当日以降に売却
                for _i, _r in df[_ind:].iterrows():
                    sell_date = df.loc[_i]['date']
                    sell_value = None
                    # その日中に set_profit 上回った場合
                    if df.loc[_i]['high'] >= ifdoco_set_profit and df.loc[_i]['open'] <= ifdoco_set_profit:
                        sell_value = ifdoco_set_profit
                        break
                    # 始値が set_profit 上回った場合
                    if df.loc[_i]['open'] > ifdoco_set_profit:
                        sell_value = df.loc[_i]['open']
                        break

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition1s.append(sell_condition1)
                sell_condition2s.append(sell_condition2)
                break

    # 後処理
    # print(len(sell_dates), len(buy_dates))
    if len(sell_dates) < len(buy_dates):
        for i in range(len(buy_dates) - len(sell_dates)):
            sell_dates.append(None)
            sell_values.append(None)
            sell_condition1s.append(None)
            sell_condition2s.append(None)

    # print(df_buy)
    # print('buy_dates', buy_dates)
    # print('buy_values', buy_values)
    # print('sell_orders', sell_orders)
    # print('sell_condition1s', sell_condition1s)
    # print('sell_condition2s', sell_condition2s)
    # print('sell_dates', sell_dates)
    # print('sell_values', sell_values)
    # print('units', units)
    # print('buy_dates', len(buy_dates))
    # print('buy_values', len(buy_values))
    # print('sell_orders', len(sell_orders))
    # print('sell_condition1s', len(sell_condition1s))
    # print('sell_condition2s', len(sell_condition2s))
    # print('sell_dates', len(sell_dates))
    # print('sell_values', len(sell_values))
    # print('units', len(units))

    df_profit = pd.DataFrame({'buy_dates': buy_dates,
                              'buy_values': buy_values,
                              'sell_orders': sell_orders,
                              'sell_condition1s': sell_condition1s,
                              'sell_condition2s': sell_condition2s,
                              'sell_dates': sell_dates,
                              'sell_values': sell_values,
                              '注文数': units})  # .dropna(how='any')

    df_profit['1株あたりの損益'] = df_profit['sell_values'] - df_profit['buy_values']
    df_profit['購入額'] = df_profit['buy_values'] * df_profit['注文数']
    df_profit['売却額'] = df_profit['sell_values'] * df_profit['注文数']
    df_profit['利益'] = df_profit['売却額'] - df_profit['購入額']

    return df_profit


def sell_pattern1(df, df_buy):
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
        df_sell: dfの購入シグナルインデックス、注文するIDOCOの利確・損切値、通常指値、通常逆指値の値を詰めたデータフレーム
    """
    df_sell = pd.DataFrame(columns=['signal_ind',
                                    'ifdoco_set_profit', 'ifdoco_stop_loss',
                                    'limit_price',
                                    'stop_loss'])
    for signal_ind in df_buy.index:

        # 購入シグナル当日の25MAとの乖離(+5～+8%)を超えた値をIFDOCO-成行の利確とする(利確)
        ifdoco_set_profit = df.loc[signal_ind]['25MA'] * 1.08  # 8%にしておく

        # 購入シグナルの出た前日or当日の安値の-5円をIFDOCO-成行の損切とする（損切）
        stop_loss1 = df.loc[signal_ind - 1]['low'] - 5
        stop_loss2 = df.loc[signal_ind]['low'] - 5
        if stop_loss1 < stop_loss2:
            ifdoco_stop_loss = stop_loss2
        else:
            ifdoco_stop_loss = stop_loss1

        # シグナル当日終値の上にある75MA、200MAの大きい方を通常-指値とする(利確)
        if (df.loc[signal_ind]['close'] <= df.iloc[signal_ind]['75MA']) \
            or (df.loc[signal_ind]['close'] <= df.iloc[signal_ind]['200MA']):

            if df.iloc[signal_ind]['75MA'] > df.iloc[signal_ind]['200MA']:
                limit_price = df.iloc[signal_ind]['75MA']
            else:
                limit_price = df.iloc[signal_ind]['200MA']
        else:
            limit_price = None

        # シグナル当日の終値の下にある5MAを通常-逆指値とする（損切）
        if df.iloc[signal_ind]['5MA'] < df.iloc[signal_ind]['close']:
            stop_loss = df.iloc[signal_ind]['5MA']
        else:
            stop_loss = None

        row = pd.Series([signal_ind, ifdoco_set_profit, ifdoco_stop_loss, limit_price, stop_loss], index=df_sell.columns)
        df_sell = df_sell.append(row, ignore_index=True)

    df_sell.index = df_sell['signal_ind'].astype(int).values
    df_sell = df_sell.drop(['signal_ind'], axis=1)
    df_sell['注文数'] = df_buy['注文数']
    df_sell['code'] = df_buy['code']
    df_sell['date'] = df_buy['date']
    # print(df_sell)
    return df_sell


def sell_pattern1_1(df, df_buy):
    """
    売り条件:
    - sell_pattern1()の条件
    - 買いの指値列をつけて指値で買う
    """
    df_sell = sell_pattern1(df, df_buy)
    df_sell['buy_limit_price'] = df_buy['buy_limit_price']
    return df_sell


def edit_df_sell_pattern1_1(df_buy, df_sell):
    """ sell_pattern1_1()のdf_buyに会うように注文条件１, 注文条件２, 利確, 損切を入れる """

    df_buy = pd.merge(df_buy, df_sell, on=['code', 'date', '注文数'])
    df_buy = df_buy.rename(columns={'buy_limit_price_x': 'buy_limit_price'})

    for _ind, _row in df_buy.fillna(0).iterrows():
        # 通常の逆指値、通常の指値、IFDOCOの損切ライン、
        stop_loss = _row['stop_loss']
        limit_price = _row['limit_price']
        ifdoco_stop_loss = _row['ifdoco_stop_loss']
        ifdoco_set_profit = _row['ifdoco_set_profit']
        # print(stop_loss, limit_price, ifdoco_stop_loss, ifdoco_set_profit)

        df_buy.at[_ind, '損切'] = ifdoco_stop_loss
        if stop_loss > ifdoco_stop_loss:
            df_buy.at[_ind, '損切'] = stop_loss

        df_buy.at[_ind, '利確'] = ifdoco_set_profit
        if 0 < limit_price < ifdoco_set_profit:
            df_buy.at[_ind, '利確'] = limit_price

        df_buy.at[_ind, '注文条件１'] = 'IFDOCO'
        df_buy.at[_ind, '注文条件２'] = '成行'

    return df_buy


def set_buy_df_for_auto(df_buy, kubun='現物'):
    """ 自動売買用のcsv列作成 買い用 """
    df_buy = df_buy.reset_index(drop=True).reset_index()
    df_buy = df_buy.rename(columns={'index': '注文番号', 'code': '証券コード'})
    df_buy['注文番号'] = df_buy['注文番号'] + 1
    df_buy['シグナル日'] = df_buy['date']
    df_buy['取引方法'] = '新規'
    df_buy['取引区分'] = kubun + '買い'
    df_buy['信用取引区分'] = ''

    if df_buy['注文条件１'].all() == '':
        df_buy['注文条件１'] = '通常'

    if df_buy['buy_limit_price'].all() != 0:
        df_buy['注文条件２'] = '指値'
        df_buy['指値'] = df_buy['buy_limit_price']

    df_buy['注文日'] = ''
    df_buy['約定日'] = ''
    df_buy = df_buy.sort_values(by=['シグナル日'])
    return df_buy


def set_sell_df_for_auto(df_sell, kubun='現物'):
    """ 自動売買用のcsv列作成 手じまい用 """
    df = pd.DataFrame()

    dates = []  # システム注文日
    codes = []  # 証券コード
    units = []  # 注文数
    sell_condition1s = []  # 注文条件１
    sell_condition2s = []  # 注文条件２
    limit_prices = []  # 指値
    set_profits = []  # 利確
    stop_losses = []  # 損切

    for ind, row in df_sell.iterrows():

        dates.append(row['date'])
        codes.append(row['code'])
        units.append(row['注文数'])
        sell_condition1s.append('IFDOCO')
        sell_condition2s.append('成行')
        limit_prices.append(None)
        set_profits.append(row['ifdoco_set_profit'])
        stop_losses.append(row['ifdoco_stop_loss'])

        if np.isnan(row['limit_price']) == False:
            dates.append(row['date'])
            codes.append(row['code'])
            units.append(row['注文数'])
            sell_condition1s.append('通常')
            sell_condition2s.append('指値')
            limit_prices.append(row['limit_price'])
            set_profits.append(None)
            stop_losses.append(None)

        if np.isnan(row['stop_loss']) == False:
            dates.append(row['date'])
            codes.append(row['code'])
            units.append(row['注文数'])
            sell_condition1s.append('通常')
            sell_condition2s.append('逆指値')
            limit_prices.append(row['stop_loss'])
            set_profits.append(None)
            stop_losses.append(None)

    df['シグナル日'] = dates
    df['注文番号'] = range(len(dates))
    df['証券コード'] = codes
    df['取引方法'] = '新規'  # '手仕舞い'
    df['取引区分'] = kubun + '売り'
    df['信用取引区分'] = ''
    df['注文数'] = units
    df['注文条件１'] = sell_condition1s
    df['注文条件２'] = sell_condition2s
    df['指値'] = limit_prices
    df['利確'] = set_profits
    df['損切'] = stop_losses
    df['注文日'] = ''
    df['約定日'] = ''
    return df.sort_values(by=['シグナル日'])


def calc_stock_index(df, profit_col='利益', deposit=1000000):
    """ 株価指標計算する """
    df = df.dropna(subset=[profit_col])
    df[profit_col] = df[profit_col].astype(float)  # 欠損ある型が変わりうまく計算できない時があったので

    if df.shape[0] > 0:
        df_win = df[df[profit_col] > 0]
        df_lose = df[df[profit_col] < 0]

        winning_percentage = (df_win.shape[0] / df.shape[0]) * 100
        print('勝率:', round(winning_percentage, 2), '%', f'(勝ち={df_win.shape[0]}回, 負け={df_lose.shape[0]}回)')

        payoff_ratio = df_win[profit_col].mean() / abs(df_lose[profit_col].mean())
        print('ペイオフレシオ（勝ちトレードの平均利益額が負けトレードの平均損失額の何倍か）:', round(payoff_ratio, 2))

        profit_factor = df_win[profit_col].sum() / abs(df_lose[profit_col].sum())
        print('プロフィットファクター（総利益が総損失の何倍か）:', round(profit_factor, 2))

        sharp_ratio = df[profit_col].mean() / df[profit_col].std()
        print('シャープレシオ（利益のばらつき）:', round(sharp_ratio, 2))

        def calc_max_drawdown(prices):
            """最大ドローダウンを計算して返す"""
            cummax_ret = prices.cummax()  # cummax関数: DataFrameまたはSeries軸の累積最大値を求める。あるindexについて、そのindexとそれより前にある全要素の最大値を求めて、その結果をそのindexに格納したSeries（またはDataFrame）を返すメソッド
            drawdown = cummax_ret - prices  # 日々の総資産額のその日までの最大値とその日の総資産額との差分
            max_drawdown_date = drawdown.idxmax()  # drawdownの中で最大の値をもつ要素のindexをidxmaxメソッドで求め、そのindexを使って最大ドローダウンの値を求めている
            return drawdown[max_drawdown_date] / cummax_ret[max_drawdown_date] * 100
        max_drawdown = calc_max_drawdown(df[profit_col] + deposit)
        print(f'種銭={deposit//10000}万円としたときの最大ドローダウン（総資産額の最大下落率）:', round(max_drawdown, 2), '%')

        return winning_percentage, payoff_ratio, profit_factor, sharp_ratio, max_drawdown
    else:
        print('勝率などは計算できませんでした')


def trade(code, start_date, end_date, buy_pattern, sell_pattern, minimum_buy_threshold, under_unit):
    """ 1銘柄について、指定の買い売り条件にマッチした株価を売買する """
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
    df['注文条件１'] = ''  # 買いの注文条件１の値
    df['注文条件２'] = ''  # 買いの注文条件２の値
    df['指値'] = ''  # 買いの指値の値
    df['利確'] = ''  # 買いの利確の値
    df['損切'] = ''  # 買いの損切の値

    # pattern関数の条件にマッチしたレコードだけ抽出（購入）
    if buy_pattern == 1:
        df_buy = df.apply(buy_pattern1, axis=1).dropna(how='any')
    elif buy_pattern == 2:
        df_buy = df.apply(buy_pattern2, axis=1).dropna(how='any')
    elif buy_pattern == 2_1:
        df_buy = df.apply(buy_pattern2_1, axis=1).dropna(how='any')
    elif buy_pattern == 3:
        df_buy = df.apply(buy_pattern3, axis=1).dropna(how='any')

    # 条件に当てはまるレコード無ければ、行もしくは列がないので損益計算しない
    if df_buy.shape[0] != 0 and df_buy.shape[1] != 0:

        # 注文数指定  最低でもunder_unit枚買う -(-4 // 3)で切り上げ
        df_buy['注文数'] = [-(-minimum_buy_threshold // (v * under_unit)) * under_unit for v in df_buy['close']]
        df_buy['注文数'] = df_buy['注文数'].astype(int)

        # print(df_buy['date'])
        # 損益計算
        if sell_pattern == 1:
            df_sell = sell_pattern1(df, df_buy)
        if sell_pattern == 1_1:
            df_sell = sell_pattern1_1(df, df_buy)

        # 損益計算
        df_profit = sell_pattern_calc_profit_loss(df, df_sell)

        # print(df_profit)
        print(f"code: {code} 利益: {round(df_profit['利益'].sum())}\n")
        df_profit['code'] = code

        # df_buy追加修正
        if buy_pattern in [1, 2, 2_1, 3]:
            df_buy = edit_df_buy_pattern2(df_buy, df_sell)
        if sell_pattern in [1_1]:
            df_buy = edit_df_sell_pattern1_1(df_buy, df_sell)

        return df_profit, df_sell, df_buy
    else:
        print(f'\ncode: {code} 売買条件に当てはまるレコードなし\n')
        return None, None, None


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default='output')
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[2914])
    ap.add_argument("-sd", "--start_date", type=str, default=None)  # '2015-01-01'
    ap.add_argument("-ed", "--end_date", type=str, default=None)
    ap.add_argument("-bp", "--buy_pattern", type=int, default=2)
    ap.add_argument("-sp", "--sell_pattern", type=int, default=1)
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000)
    ap.add_argument("-uu", "--under_unit", type=int, default=100)
    ap.add_argument("-dep", "--deposit", type=int, default=1000000, help="deposit money.")
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

    # 指定銘柄についてシュミレーション
    df_profits = None
    df_sells = None
    df_buys = None
    for code in tqdm(codes):
        df_profit, df_sell, df_buy = trade(code, start_date, end_date,
                                           args['buy_pattern'], args['sell_pattern'],
                                           args['minimum_buy_threshold'], args['under_unit'])
        if df_profits is None:
            if df_profit is not None:
                df_profits = df_profit
                df_sells = df_sell
                df_buys = df_buy
        else:
            if df_profit is not None:
                df_profits = pd.concat([df_profits, df_profit], ignore_index=True)
                df_sells = pd.concat([df_sells, df_sell], ignore_index=True)
                df_buys = pd.concat([df_buys, df_buy], ignore_index=True)

    # シュミレーション結果あれば後続処理続ける
    if df_profits is not None:

        # シグナル日の確認用にcsvファイル一応出しておく
        df_buys.to_csv(os.path.join(output_dir, 'signal_rows.csv'), index=False, encoding='shift-jis')

        # 1日の種銭をdepositとして予算内で買えるものだけに絞る
        if args['deposit'] != 0:
            print(f"INFO: 1日の種銭を{args['deposit']//10000}万円として予算内で買えるものだけに絞る")
            df_profits_deposit = pd.DataFrame(columns=df_profits.columns)

            for buy_date, df_date in df_profits.groupby(['buy_dates']):
                _deposit = args['deposit']
                for index, row in df_date.iterrows():
                    if _deposit - row['購入額'] < 0:
                        break
                    else:
                        df_profits_deposit = pd.concat([df_profits_deposit, pd.DataFrame([row])], ignore_index=True)
                        _deposit = _deposit - row['購入額']
            df_profits = df_profits_deposit

        # 損益確認用csv出力
        df_profits = df_profits.sort_values(by=['buy_dates', 'sell_dates', 'code'])
        df_profits = df_profits[['code', 'buy_dates', 'buy_values',
                                 'sell_orders',
                                 'sell_condition1s', 'sell_condition2s',
                                 'sell_dates', 'sell_values',
                                 '1株あたりの損益', '注文数', '購入額', '売却額', '利益']]  # 'sell_order0s',
        df_profits.to_csv(os.path.join(output_dir, 'simple_trade_v2.csv'), index=False, encoding='shift-jis')
        print('総利益:', round(df_profits['利益'].sum(), 2))

        # 株価指標計算
        print()
        _ = calc_stock_index(df_profits, profit_col='利益', deposit=args['deposit'])

        # 自動売買用csvの列作成
        # print(df_buys.columns)
        df_buy_for_auto = set_buy_df_for_auto(df_buys, kubun=args['kubun'])
        # df_sell_for_auto = set_sell_df_for_auto(df_sells, kubun=args['kubun'])
        # df_for_autos = pd.concat([df_buy_for_auto, df_sell_for_auto], ignore_index=True)
        df_for_autos = df_buy_for_auto.reset_index(drop=True)

        # 自動売買用csv出力
        df_for_autos = df_for_autos.dropna(subset=['シグナル日'])
        df_for_autos = df_for_autos.sort_values(by=['取引区分'], ascending=False)
        df_for_autos = df_for_autos.sort_values(by=['シグナル日', '証券コード'])
        df_for_autos['注文番号'] = range(1, df_for_autos.shape[0] + 1)
        df_for_autos = df_for_autos[['シグナル日', '注文番号', '証券コード', '取引方法', '取引区分', '信用取引区分',
                                    '注文数', '注文条件１', '注文条件２', '指値', '利確', '損切', '注文日', '約定日']]
        df_for_autos.to_csv(os.path.join(output_dir, 'auto_order.csv'), index=False, encoding='shift-jis')

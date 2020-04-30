#!/usr/bin/env python
# coding: utf-8
"""
1銘柄について、指定の株価抽出条件にマッチした株価からn日後に売買したときの損益を計算する
自動売買で使うcsvも出す
Usage:
    $ activate stock
    $ python ./simple_trade_v2_for_auto.py -cs 7013 -bp 2
    $ python ./simple_trade_v2_for_auto.py -cs 2914 7013 8304 -bp 2
    $ python ./simple_trade_v2_for_auto.py -cs 7013 8304 3407 2502 2802 4503 6857 6113 6770 8267 7202 5019 8001 4208 4523 5201 9202 6472 9613 9437 6361 8725 2413 6103 9532 3861 4578 1802 6703 9007 6645 7733 4452 6952 1812 9107 7012 9503 2801 7751 6971 4151 2503 6326 3405 8253 9008 9009 9433 5406 1605 9766 4902 6301 1721 7186 4751 2501 3436 6674 5411 5020 6473 3086 4507 8355 4911 7762 1803 9104 4004 4063 8303 5401 9412 7735 7269 7270 5232 4005 5713 6302 8053 5802 8830 6724 1928 9735 4689 3382 2768 6758 8729 9984 8630 4568 8750 6367 1801 7912 4506 5541 5233 6976 1925 8601 8233 2531 4502 8331 4519 9502 8795 2432 3401 6762 4631 4543 4061 6902 4324 5301 9022 3289 8035 8766 9531 9005 8804 9501 4042 5332 9001 9602 5707 5901 3101 3402 5714 4043 7911 7203 8015 4704 7731 9021 2871 1963 4021 7201 2002 3105 6988 5202 5333 4272 5703 1332 6471 3863 5631 4041 2914 9062 6701 5214 9432 2282 6178 9101 8604 1808 6752 7832 9020 6305 6501 7004 7205 9983 6954 8028 8354 5803 6702 6504 4901 5108 5801 7267 8628 7261 8252 1333 8002 8411 7003 4183 5706 8309 8316 8031 8801 3099 4188 7211 7011 8058 9301 8802 6503 5711 8306 6479 2269 6506 9064 7951 7272 3103 6841 5101 6098 7752 8308
"""
import argparse
import datetime
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


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
        # - 当日高値＋１で逆指値
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


def sell_pattern1(df, df_buy):
    """
    売り条件:
    - シグナルの出た（前日or当日）安値の-5円で逆指値売却（損切）
    - シグナル当日の25MAとの乖離(+5～+8%)を超えた値で逆指値売却(利確)
    - シグナル当日終値の上にある75MA、200MAに最新データがタッチ → 翌日寄付売却（利確）
    - 陰線の終値で最新データが5MAを(0.3～0.5)%割込む → 翌日寄付売却（損切）
    """
    # df['date'] = pd.to_datetime(df['date'])
    buy_dates, buy_values = [], []
    sell_condition2s, sell_dates, sell_values, stop_losses, set_profits = [], [], [], [], []

    for signal_ind in df_buy.index:
        buy_dates.append(df.loc[signal_ind + 1]['date'])
        buy_values.append(df.loc[signal_ind + 1]['open'])

        # シグナルの出た（前日or当日）安値の-5円で逆指値売却（損切）
        stop_loss1 = df.loc[signal_ind - 1]['low'] - 5
        stop_loss2 = df.loc[signal_ind]['low'] - 5
        if stop_loss1 < stop_loss2:
            stop_loss = stop_loss2
        else:
            stop_loss = stop_loss1
        stop_losses.append(stop_loss)

        # シグナル当日の25MAとの乖離(+5～+8%)を超えた値で指値売却(利確)
        set_profit = df.loc[signal_ind]['25MA'] * 1.08
        set_profits.append(set_profit)

        # シグナルの次の日（購入日）からデータフレームループ
        for sell_sig, series in df[signal_ind + 1: -1].iterrows():
            sell_date = None
            sell_value = None
            sell_condition2 = None

            if series['low'] <= stop_loss:
                sell_condition2 = '逆指値'  # 損切
                sell_date = df.loc[sell_sig]['date']  # 即日損切
                sell_value = stop_loss

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition2s.append(sell_condition2)
                break

            if df.loc[signal_ind]['close'] < set_profit <= series['high']:
                sell_condition2 = '指値'  # 利確
                sell_date = df.loc[sell_sig]['date']  # 即日利確
                sell_value = set_profit

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition2s.append(sell_condition2)
                break

            # シグナル当日終値の上にある75MA、200MAに最新データがタッチ → 翌日寄付売却（利確）
            if (df.loc[signal_ind]['close'] <= df.iloc[signal_ind]['75MA'] <= series['high']) \
                or (df.loc[signal_ind]['close'] <= df.iloc[signal_ind]['200MA'] <= series['high']):
                sell_condition2 = '成行'  # 利確
                sell_date = df.loc[sell_sig + 1]['date']  # 翌日利確
                sell_value = df.loc[sell_sig + 1]['open']

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition2s.append(sell_condition2)
                break

            # 陰線の終値で最新データが5MAを(0.3～0.5)%割込む → 翌日寄付売却（損切）
            if df.iloc[signal_ind]['5MA'] < df.iloc[signal_ind]['close'] \
                and series['5MA'] >= series['close']:
                sell_condition2 = '成行'  # 損切
                sell_date = df.loc[sell_sig + 1]['date']  # 翌日損切
                sell_value = df.loc[sell_sig + 1]['open']

                sell_dates.append(sell_date)
                sell_values.append(sell_value)
                sell_condition2s.append(sell_condition2)
                break

        # 後処理
        if len(sell_dates) < len(buy_dates):
            for i in range(len(buy_dates) - len(sell_dates)):
                sell_dates.append(None)
                sell_values.append(None)
                sell_condition2s.append(None)

    df_profit = pd.DataFrame({'buy_dates': buy_dates,
                              'buy_values': buy_values,
                              'sell_condition2s': sell_condition2s,
                              'sell_dates': sell_dates,
                              'sell_values': sell_values,
                              'stop_losses': stop_losses,
                              'set_profits': set_profits,
                              '注文数': df_buy['注文数']})  #.dropna(how='any')

    df_profit['1株あたりの損益'] = df_profit['sell_values'] - df_profit['buy_values']
    df_profit['購入額'] = df_profit['buy_values'] * df_profit['注文数']
    df_profit['売却額'] = df_profit['sell_values'] * df_profit['注文数']
    df_profit['利益'] = df_profit['売却額'] - df_profit['購入額']
    return df_profit


def set_buy_df_for_auto(df_buy, kubun='現物'):
    """ 自動売買用のcsv列作成 買い用 """
    df_buy = df_buy.reset_index(drop=True).reset_index()
    df_buy = df_buy.rename(columns={'index': '注文番号', 'code': '証券コード'})
    df_buy['注文番号'] = df_buy['注文番号'] + 1
    df_buy['注文実行日'] = df_buy['date']
    df_buy['取引方法'] = '新規'
    df_buy['取引区分'] = kubun + '買い'
    df_buy['信用取引区分'] = ''
    df_buy['注文条件１'] = '通常'
    df_buy['注文条件２'] = '成行'
    df_buy['指値'] = ''
    df_buy['利確'] = ''
    df_buy['損切'] = ''
    df_buy['注文日'] = ''
    df_buy['約定日'] = ''
    df_buy = df_buy.sort_values(by=['注文実行日'])
    return df_buy


def set_sell_df_for_auto(df_profit, kubun='現物'):
    """ 自動売買用のcsv列作成 手じまい用 """
    df_profit = df_profit.reset_index(drop=True).reset_index()
    df_profit = df_profit.rename(columns={'index': '注文番号', 'code': '証券コード'})
    df_profit['注文番号'] = df_profit['注文番号'] + 1
    df_profit['取引方法'] = '新規'  # '手仕舞い'
    df_profit['取引区分'] = kubun + '売り'
    df_profit['信用取引区分'] = ''

    _order_expects = []  # 注文実行日
    _sell_condition1s = []  # 注文条件１
    _sell_condition2s = []  # 注文条件２
    _sell_values = []  # 指値
    _set_profits = []  # 利確
    _stop_losses = []  # 損切
    for row in df_profit.itertuples(index=False):

        if row.sell_condition2s is not None:
            _order_expects.append(row.sell_dates)

            if row.sell_condition2s == '成行':
                _sell_condition2s.append(row.sell_condition2s)
                _sell_values.append(None)

            if row.sell_condition2s == '指値':
                _sell_condition2s.append(row.sell_condition2s)
                _sell_values.append(row.sell_values)

            if row.sell_condition2s == '逆指値':
                _sell_condition2s.append(row.sell_condition2s)
                _sell_values.append(row.sell_values)

            if row.stop_losses < row.buy_values < row.set_profits:
                _sell_condition1s.append('IFDOCO')
                _set_profits.append(row.set_profits)
                _stop_losses.append(row.stop_losses)
            else:
                _sell_condition1s.append('通常')
                _set_profits.append(None)
                _stop_losses.append(None)
        else:
            _order_expects.append(None)
            _sell_condition1s.append(None)
            _sell_condition2s.append(None)
            _sell_values.append(None)
            _set_profits.append(None)
            _stop_losses.append(None)

    df_profit['注文実行日'] = _order_expects
    df_profit['注文条件１'] = _sell_condition1s
    df_profit['注文条件２'] = _sell_condition2s
    df_profit['指値'] = _sell_values
    # df_profit['指値'] = df_profit['指値'].astype(int)
    df_profit['利確'] = _set_profits
    # df_profit['利確'] = df_profit['利確'].astype(int)
    df_profit['損切'] = _stop_losses
    # df_profit['損切'] = df_profit['損切'].astype(int)
    df_profit['注文日'] = ''
    df_profit['約定日'] = ''
    df_profit = df_profit.sort_values(by=['注文実行日'])
    return df_profit


def calc_stock_index(df, profit_col='_利益', deposit=1000000):
    """ 株価指標計算する """
    df_win = df[df[profit_col] > 0]
    df_lose = df[df[profit_col] < 0]

    winning_percentage = df_win.shape[0] / df.shape[0]
    print('勝率:', round(winning_percentage, 2), '%')

    payoff_ratio = abs(df_win[profit_col].mean() / df_lose[profit_col].mean())
    print('ペイオフレシオ（勝ちトレードの平均利益額が負けトレードの平均損失額の何倍か）:', round(payoff_ratio, 2))

    profit_factor = abs(df_win[profit_col].sum() / df_lose[profit_col].sum())
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


def trade(code, start_date, buy_pattern, sell_pattern, minimum_buy_threshold, under_unit):
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
    df['high_shift_1'] = df['high'].shift(-1).fillna(0)  # 翌日高値

    # pattern関数の条件にマッチしたレコードだけ抽出
    if buy_pattern == 1:
        df_buy = df.apply(buy_pattern1, axis=1).dropna(how='any')
    elif buy_pattern == 2:
        df_buy = df.apply(buy_pattern2, axis=1).dropna(how='any')
    elif buy_pattern == 3:
        df_buy = df.apply(buy_pattern3, axis=1).dropna(how='any')

    # 条件に当てはまるレコード無ければ、行もしくは列がないので損益計算しない
    if df_buy.shape[0] != 0 and df_buy.shape[1] != 0:

        # 注文数指定  最低でもunder_unit枚買う -(-4 // 3)で切り上げ
        df_buy['注文数'] = [-(-minimum_buy_threshold // (v * under_unit)) * under_unit for v in df_buy['close']]
        df_buy['注文数'] = df_buy['注文数'].astype(int)

        # 損益計算
        if sell_pattern == 1:
            df_profit = sell_pattern1(df, df_buy)

        # print(df_profit)
        print(f"code: {code} 利益: {round(df_profit['利益'].sum())}\n")

        df_profit['code'] = code
        return df_profit, df_buy
    else:
        print(f'\ncode: {code} 売買条件に当てはまるレコードなし\n')
        return None


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default='output')
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[7267])
    ap.add_argument("-s", "--start_date", type=str, default='2015-01-01')
    ap.add_argument("-bp", "--buy_pattern", type=int, default=2)
    ap.add_argument("-sp", "--sell_pattern", type=int, default=1)
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000)
    ap.add_argument("-uu", "--under_unit", type=int, default=100)
    ap.add_argument("-dep", "--deposit", type=int, default=1000000, help="deposit money.")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    # code = 7267  # リクルート
    # code = 6981  # 村田
    # code = 2914  # JT
    # code = 6758  # Sony
    codes = args['codes']

    # start_date = '2017-12-31'

    output_dir = args['output_dir']

    df_profit_codes = None
    df_for_autos = None
    for code in tqdm(codes):
        df_profit, df_buy = trade(code, args['start_date'], args['buy_pattern'], args['sell_pattern'],
                                  args['minimum_buy_threshold'], args['under_unit'])
        if df_profit_codes is None:
            if df_profit is not None:
                df_profit_codes = df_profit

                # 自動売買用のcsv列作成
                df_buy_for_auto = set_buy_df_for_auto(df_buy)
                # print(df_buy_for_auto)
                df_profit_for_auto = set_sell_df_for_auto(df_profit)
                # print(df_profit_for_auto)
                df_for_autos = pd.concat([df_buy_for_auto, df_profit_for_auto], ignore_index=True)

        else:
            if df_profit is not None:
                df_profit_codes = pd.concat([df_profit_codes, df_profit], ignore_index=True)

                # 自動売買用のcsv列作成
                df_buy_for_auto = set_buy_df_for_auto(df_buy)
                df_profit_for_auto = set_sell_df_for_auto(df_profit)
                df_for_auto = pd.concat([df_buy_for_auto, df_profit_for_auto], ignore_index=True)
                df_for_autos = pd.concat([df_for_autos, df_for_auto], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)

    # 損益確認用csv
    df_profit_codes = df_profit_codes.sort_values(by=['buy_dates', 'sell_dates', 'code'])
    df_profit_codes = df_profit_codes[['code', 'buy_dates', 'buy_values',
                                       'sell_condition2s', 'sell_dates', 'sell_values',
                                       '1株あたりの損益', '注文数', '購入額', '売却額', '利益']]
    df_profit_codes.to_csv(os.path.join(output_dir, 'simple_trade_v2.csv'), index=False, encoding='shift-jis')
    print('総利益:', round(df_profit_codes['利益'].sum(), 2))

    # 株価指標計算
    print()
    _ = calc_stock_index(df_profit_codes, profit_col='利益', deposit=args['deposit'])

    # 自動売買用csv
    df_for_autos = df_for_autos.dropna(subset=['注文実行日'])
    df_for_autos = df_for_autos.sort_values(by=['注文実行日'])
    df_for_autos['注文番号'] = range(1, df_for_autos.shape[0] + 1)
    df_for_autos = df_for_autos[['注文実行日', '注文番号', '証券コード', '取引方法', '取引区分', '信用取引区分',
                                 '注文数', '注文条件１', '注文条件２', '指値', '利確', '損切', '注文日', '約定日']]
    df_for_autos.to_csv(os.path.join(output_dir, 'auto_order.csv'), index=False, encoding='shift-jis')
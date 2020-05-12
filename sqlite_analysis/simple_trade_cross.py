#!/usr/bin/env python
# coding: utf-8
"""
複数銘柄について、指定の買い売り条件にマッチした株価を売買したときの損益を計算する
Usage:
    $ activate stock
    $ python ./simple_trade_cross.py
    $ python ./simple_trade_cross.py -is_new # 出力するcsvにはJTの最新の十字線のレコードだけ
    $ python ./simple_trade_cross.py -sd 2010-01-01 -ed 2020-05-10  # 期間指定
    $ python ./simple_trade_cross.py -is_csv  # csvファイルから株価データロード
    $ python ./simple_trade_cross.py -is_day  # デイトレ
    $ python ./simple_trade_cross.py -cs 7013 8304 3407 2502 2802 4503 6857 6113 6770 8267 7202 5019 8001 4208 4523 5201 9202 6472 9613 9437 6361 8725 2413 6103 9532 3861 4578 1802 6703 9007 6645 7733 4452 6952 1812 9107 7012 9503 2801 7751 6971 4151 2503 6326 3405 8253 9008 9009 9433 5406 1605 9766 4902 6301 1721 7186 4751 2501 3436 6674 5411 5020 6473 3086 4507 8355 4911 7762 1803 9104 4004 4063 8303 5401 9412 7735 7269 7270 5232 4005 5713 6302 8053 5802 8830 6724 1928 9735 4689 3382 2768 6758 8729 9984 8630 4568 8750 6367 1801 7912 4506 5541 5233 6976 1925 8601 8233 2531 4502 8331 4519 9502 8795 2432 3401 6762 4631 4543 4061 6902 4324 5301 9022 3289 8035 8766 9531 9005 8804 9501 4042 5332 9001 9602 5707 5901 3101 3402 5714 4043 7911 7203 8015 4704 7731 9021 2871 1963 4021 7201 2002 3105 6988 5202 5333 4272 5703 1332 6471 3863 5631 4041 2914 9062 6701 5214 9432 2282 6178 9101 8604 1808 6752 7832 9020 6305 6501 7004 7205 9983 6954 8028 8354 5803 6702 6504 4901 5108 5801 7267 8628 7261 8252 1333 8002 8411 7003 4183 5706 8309 8316 8031 8801 3099 4188 7211 7011 8058 9301 8802 6503 5711 8306 6479 2269 6506 9064 7951 7272 3103 6841 5101 6098 7752 8308
    $ python ./simple_trade_cross.py -all  # 最近のデータだけで全銘柄実行
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
import traceback


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
        # 200MA計算するためにだいぶ前から取る
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date() - datetime.timedelta(days=500)
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


def load_stock_csv(code, csv_path, start_date, end_date):
    df = pd.read_csv(csv_path, encoding='shift-jis',
                     dtype={1: float, 2: float, 3: float, 4: float, 5: float, 6: float},
                     parse_dates=[0])
    # 200MA計算するためにだいぶ前から取る
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date() - datetime.timedelta(days=500)
    df = df[(df['日付'] >= start_date) & (df['日付'] <= end_date)]
    df = df.rename(columns={'日付': 'date', '始値': 'open', '高値': 'high', '安値': 'low', '終値': 'close', '出来高': 'volume'})
    df['code'] = code
    return df.reset_index(drop=True)


def add_option_columns(df, start_date):
    """ 株価データフレームに列追加 """
    df['5MA'] = df['close'].rolling(window=5).mean()
    df['25MA'] = df['close'].rolling(window=25).mean()
    df['75MA'] = df['close'].rolling(window=75).mean()
    df['200MA'] = df['close'].rolling(window=200).mean()
    df['next_open_current_close_diff'] = df['open'].shift(-1).fillna(0) - df['close']  # 翌日始値-当日終値
    df['next_open_current_high_diff'] = df['open'].shift(-1).fillna(0) - df['high']  # 翌日始値-当日高値
    df['open_rate'] = (abs(df['open'] - df['low']) / abs(df['high'] - df['low']))  # 始値-安値の絶対値/高値-安値の絶対値
    df['close_rate'] = (abs(df['close'] - df['low']) / abs(df['high'] - df['low']))  # 終値-安値の絶対値/高値-安値の絶対値

    # start_dateからのレコードだけにする（200MA出すためにstart_date より前のレコード保持しているため）
    df = df.dropna(subset=['200MA'])
    df['date_datetime'] = pd.to_datetime(df['date'])
    df = df.query(f'date_datetime >= "{start_date}"')
    df = df.reset_index(drop=True)

    return df


def buy_pattern_cross(row):
    """
    買い条件:
    - 当日に十字線が出た。十字線の定義はその日の値幅(高値-安値の絶対値)を10として4から6に始値と終値がある
    - 翌日の始値が前日の終値より高い
    """
    if abs(row['close_rate'] - row['open_rate']) <= 0.2 \
        and 0.4 <= row['close_rate'] <= 0.6 \
        and 0.4 <= row['open_rate'] <= 0.6 \
        and row['next_open_current_close_diff'] > 0.0:
        return row
    else:
        return pd.Series()


class CrossTrade():
    """ 十字線をシグナルとし、日またいでトレード """
    def __init__(self, deposit=1000000, unit=100):
        self.deposit = deposit  # 現在の預り金
        self.unit = unit  # 1回の購入株数

    def search_stop_loss(self, row, stop_loss: float):
        """ 損切したときの株価探し、損切時の株価を返す """
        if row['open'] <= stop_loss:
            sell = row['open']
        elif row['low'] <= stop_loss <= row['high']:
            sell = stop_loss
        else:
            sell = None
        return sell

    def search_set_profit(self, row, set_profit: float):
        """ 利確したときの株価探し、利確時の株価を返す """
        if row['open'] >= set_profit:
            sell = row['open']
        elif row['low'] <= set_profit <= row['high']:
            sell = set_profit
        else:
            sell = None
        return sell

    def execute_sell(self, row, buy: float, set_profit: float, stop_loss: float):
        """ 損切/利確実行して損益計算する。売買時のレコード返す"""
        # 損切
        sell = self.search_stop_loss(row, stop_loss)
        if sell is not None:
            self.deposit = self.deposit + sell * self.unit
            print(f"損切: {row['date']}, {int(buy)}→{int(sell)}円/1株,　所持金={int(self.deposit)}円")
            row['sell'] = sell
            return row

        # 利確
        sell = self.search_set_profit(row, set_profit)
        if sell is not None:
            self.deposit = self.deposit + sell * self.unit
            print(f"利確: {row['date']}, {int(buy)}→{int(sell)}円/1株,　所持金={int(self.deposit)}円")
            row['sell'] = sell
            return row

        return pd.Series()

    def trade(self, signal_row, df):
        """ 株購入→売却を行う """
        # シグナルの翌日からのレコードだけ取得
        df_signal_next_day = df[signal_row.name + 1:]

        # 株購入。シグナルの翌日始値で購入
        buy = df_signal_next_day.iloc[0]['open']
        self.deposit = self.deposit - buy * self.unit

        # ロスカットは十字線の安値-1円
        stop_loss = signal_row['low'] - 1.0

        # 利確は十字線の終値*20%とする
        set_profit = signal_row['close'] * 1.2

        # 損切/利確
        for index, row in df_signal_next_day.iterrows():
            row = self.execute_sell(row, buy, set_profit, stop_loss)
            if row.empty == False:
                row['buy_date'] = df_signal_next_day.iloc[0]['date']
                row['buy'] = buy
                return row
        return row


def buy_pattern_cross_v2(row):
    """
    買い条件:
    - 当日に十字線が出た。十字線の定義はその日の値幅(高値-安値の絶対値)を10として4から6に始値と終値がある
    - 翌日の始値が前日の高値より高い
    - 当日終値が75MAより高い
    - 当日終値が200MAより高い
    """
    if abs(row['close_rate'] - row['open_rate']) <= 0.2 \
        and 0.4 <= row['close_rate'] <= 0.6 \
        and 0.4 <= row['open_rate'] <= 0.6 \
        and row['next_open_current_high_diff'] > 0.0 \
        and row['75MA'] <= row['close'] \
        and row['200MA'] <= row['close']:
        return row
    else:
        return pd.Series()


class CrossTradeDay():
    """ 十字線をシグナルとし、デイトレード """
    def __init__(self, deposit=1000000, unit=100):
        self.deposit = deposit  # 現在の預り金
        self.unit = unit  # 1回の購入株数

    def search_stop_loss(self, row, stop_loss: float):
        """ 損切したときの株価探し、損切時の株価を返す """
        if row['open'] <= stop_loss:
            sell = row['open']
        elif row['low'] <= stop_loss <= row['high']:
            sell = stop_loss
        else:
            sell = None
        return sell

    def execute_sell(self, row, buy: float, set_profit: float, stop_loss: float):
        """ 利確/損切実行して損益計算する。売買時のレコード返す"""
        # 損切
        sell = self.search_stop_loss(row, stop_loss)
        if sell is not None:
            self.deposit = self.deposit + sell * self.unit
            row['sell'] = sell
            return row

        # デイトレなので損切なければ終値で利確
        sell = set_profit
        self.deposit = self.deposit + sell * self.unit
        row['sell'] = sell

        return row

    def trade(self, signal_row, df):
        """ 株購入→売却を行う """
        # シグナルの翌日からのレコードだけ取得
        df_signal_next_day = df[signal_row.name + 1: ]

        # 株購入。シグナルの翌日始値で購入
        buy = df_signal_next_day.iloc[0]['open']
        self.deposit = self.deposit - buy * self.unit

        # ロスカットは十字線の安値-1円
        stop_loss = signal_row['low'] - 1.0

        # 利確は株購入日の終値とする
        set_profit = df_signal_next_day.iloc[0]['close']

        # 利確/損切
        row = df_signal_next_day.iloc[0]
        row = self.execute_sell(row, buy, set_profit, stop_loss)
        row['buy_date'] = row['date']
        row['buy'] = buy

        return row


def test_func():
    """
    テスト駆動開発でのテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """
    # SqlliteMgr + add_option_columns + buy_pattern_cross()
    start_date = '2017-09-20'
    df = SqlliteMgr().get_code_price('1878', start_date, '2017-09-25')
    df = add_option_columns(df, start_date)
    df_signal = df.apply(buy_pattern_cross, axis=1).dropna(how='all')[df.columns]
    assert df_signal.shape[0] == 1  # 必ず1件だけ取れるはず

    # load_stock_csv + add_option_columns + buy_pattern_cross()
    start_date = '2017-09-20'
    csv_path = os.path.join(r'D:\DB_Browser_for_SQLite\csvs\kabuoji3', '1878' + '.csv')
    df = load_stock_csv('1878', csv_path, start_date, '2017-09-25')
    df = add_option_columns(df, start_date)
    df_signal = df.apply(buy_pattern_cross, axis=1).dropna(how='all')[df.columns]
    assert df_signal.shape[0] == 1  # 必ず1件だけ取れるはず

    # CrossTrade + buy_pattern_cross
    start_date = '2018-01-01'
    df = SqlliteMgr().get_code_price('4507', start_date, '2020-05-10')
    df = add_option_columns(df, start_date)
    df_signal = df.apply(buy_pattern_cross, axis=1).dropna(how='all')[df.columns]
    crosstrade = CrossTrade()
    df_trade = df_signal.apply(crosstrade.trade, df=df, axis=1).dropna(how='all')
    df_trade = df_trade[[*df_signal.columns.to_list(), *['buy_date', 'buy', 'sell']]]
    print(df_trade.tail())
    assert df_trade.shape[0] == 9  # 必ず9件だけ取れるはず

    # CrossTradeDay + buy_pattern_cross_v2
    start_date = '2018-01-01'
    df = SqlliteMgr().get_code_price('4507', start_date, '2020-05-10')
    df = add_option_columns(df, start_date)
    df_signal = df.apply(buy_pattern_cross_v2, axis=1).dropna(how='all')[df.columns]
    crosstrade = CrossTradeDay()
    df_trade = df_signal.apply(crosstrade.trade, df=df, axis=1).dropna(how='all')
    df_trade = df_trade[[*df_signal.columns.to_list(), *['buy_date', 'buy', 'sell']]]
    print(df_trade.tail())
    assert df_trade.shape[0] == 1  # 必ず1件だけ取れるはず


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default='output')
    ap.add_argument("-cd", "--csv_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabuoji3')  # 株価csvの格納ディレクトリ
    ap.add_argument("-cs", "--codes", type=int, nargs='*', default=[2914])
    ap.add_argument("-sd", "--start_date", type=str, default=None)  # '2015-01-01'
    ap.add_argument("-ed", "--end_date", type=str, default=None)
    ap.add_argument("-p", "--pattern", type=int, default=2_1)
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000)
    ap.add_argument("-uu", "--under_unit", type=int, default=100)
    ap.add_argument("-dep", "--deposit", type=int, default=1000000, help="deposit money.")
    ap.add_argument("-all", "--is_all_codes", action='store_const', const=True, default=False, help="all stock flag.")
    ap.add_argument("-is_csv", "--is_csv_load", action='store_const', const=True, default=False, help="load stock csv path flag.")
    ap.add_argument("-is_day", "--is_day_trade", action='store_const', const=True, default=False, help="CrossTradeDay flag.")
    ap.add_argument("-is_new", "--is_leave_new_only", action='store_const', const=True, default=False, help="leave current date record flag.")
    return vars(ap.parse_args())


def run_trade(df, code, start_date, is_day_trade):
    try:
        df = add_option_columns(df, start_date)
        if is_day_trade:
            df_signal = df.apply(buy_pattern_cross_v2, axis=1).dropna(how='all')[df.columns]
            crosstrade = CrossTradeDay()
        else:
            df_signal = df.apply(buy_pattern_cross, axis=1).dropna(how='all')[df.columns]
            crosstrade = CrossTrade()

        df_trade = df_signal.apply(crosstrade.trade, df=df, axis=1).dropna(how='all')
        df_trade = df_trade[[*df_signal.columns.to_list(), *['buy_date', 'buy', 'sell']]]

        win_count = df_trade[df_trade['sell'] > df_trade['buy']].shape[0]
        total_count = df_trade.shape[0]
        result_profit = (df_trade['sell'] * 100 - df_trade['buy'] * 100).sum()
        print(f"code: {code}, result_profit: {int(result_profit)} 円, " +
              f"win_rate: {round(win_count / total_count, 3)} ({win_count}/{total_count})")
        return df_trade
    except Exception:
        # traceback.print_exc()
        return None


def main(args):
    # code = 7267  # リクルート
    # code = 6981  # 村田
    # code = 2914  # JT
    # code = 6758  # Sony
    codes = [int(pathlib.Path(p).stem) for p in glob.glob(r'D:\DB_Browser_for_SQLite\csvs\kabuoji3\*.csv')] \
        if args['is_all_codes'] else args['codes']

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 今日の日付
    end_date = datetime.datetime.today().strftime("%Y-%m-%d") \
        if args['end_date'] is None else args['end_date']

    start_date = args['start_date']
    if start_date is None:
        start_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date() - datetime.timedelta(days=500)  # 500日前からスタート
        start_date = start_date.strftime("%Y-%m-%d")
    print('start_date, end_date', start_date, end_date)

    df_concat = None
    pbar = tqdm(codes)
    for code in pbar:
        pbar.set_description(str(code))

        # csvファイルからデータロードする場合 else DBからデータロードする場合
        df = load_stock_csv(code, os.path.join(args['csv_dir'], str(code) + '.csv'), start_date, end_date) \
            if args['is_csv_load'] else SqlliteMgr().get_code_price(code, start_date, end_date)

        df_trade = run_trade(df, code, start_date, args['is_day_trade'])

        df_concat = df_trade \
            if df_concat is None else pd.concat([df_concat, df_trade], ignore_index=True)

    if df_concat is not None:
        win_count = df_concat[df_concat['sell'] > df_concat['buy']].shape[0]
        total_count = df_concat.shape[0]
        result_profit = (df_concat['sell'] * 100 - df_concat['buy'] * 100).sum()
        print(f'### result_profit_sum: {int(result_profit)} 円 ###')
        print(f"### win_rate: {round(win_count / total_count, 3)} ({win_count}/{total_count}) ###")

        if args['is_leave_new_only']:
            df_concat['date'] = pd.to_datetime(df_concat['date'])
            run_day = df_concat['date'].iloc[-1]  # 最後の日だけ残す
            df_concat = df_concat[df_concat['date'] >= run_day]
        else:
            pass

        df_concat = df_concat.sort_values(by=['date', 'code'])  # 日付,銘柄コードの昇順にしておく
        df_concat.to_csv(os.path.join(output_dir, 'simple_trade_cross.csv'), index=False, encoding='shift-jis')
    else:
        print('該当データなし')


if __name__ == '__main__':
    main(get_args())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1銘柄を毎月積み立てたときをsimulator.pyでテストする(buy_and_hold.pyの種銭毎月増やして毎月買って積立てる版)
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter3/nikkei_tsumitate_trade.py

＜シミュレーションの条件＞
・毎月始めに所持金で買えるだけ指定の1銘柄を買う（始値で買う）。
・毎月所持金増やす。
・最終日にその銘柄を全数始値で売る。

Usage:
    $ activate stock
    $ python tsumitate_trade_plus_alpha.py
"""
import datetime
import argparse
import sqlite3
import numpy as np
import pandas as pd

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim
# from golden_core30 import create_stock_data


def pattern2(row):
    """
    追加の株価購入条件:
    - 終値が5MA,25MAより上
    - 終値が前日からの直近10日間の最大高値より上
    - 出来高が前日比30%以上増
    - 終値が当日の値幅の上位70%以上(大陽線?)
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
        # return row
        return 1
    else:
        # return pd.Series()
        return 0


def create_stock_data(db_file_name, code_list, start_date, end_date):
    """指定した銘柄(code_list)それぞれの単元株数と日足(始値・終値 etc）を含む辞書を作成
    """
    stocks = {}
    tse_index = sim.tse_date_range(start_date, end_date)
    conn = sqlite3.connect(db_file_name)
    for code in code_list:
        unit = conn.execute('SELECT unit from brands WHERE code = ?',
                            (code,)).fetchone()[0]
        prices = pd.read_sql('SELECT * '
                             'FROM prices '
                             'WHERE code = ? AND date BETWEEN ? AND ?'
                             'ORDER BY date',
                             conn,
                             params=(code, start_date, end_date),
                             parse_dates=('date',),
                             index_col='date')
        # print(prices)

        # ### plu_alpha #### #
        prices['5MA'] = prices['close'].rolling(window=5).mean()
        prices['25MA'] = prices['close'].rolling(window=25).mean()
        prices['close_10MAX'] = prices['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
        prices['high_10MAX'] = prices['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
        prices['high_shfi1_10MAX'] = prices['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
        # prices['high_shfi1_15MAX'] = prices['high'].shift(1).fillna(0).rolling(window=15, min_periods=0).max()  # 前日からの直近15日間の中で最大高値
        prices['volume_1diff_rate'] = (prices['volume'] - prices['volume'].shift(1).fillna(0)) / prices['volume'].shift(1).fillna(0)  # 前日比出来高
        prices['open_close_1diff'] = prices['open'].shift(-1).fillna(0) - prices['close']  # 翌日始値-当日終値
        # 購入フラグ付ける
        prices['buy_flag'] = prices.apply(pattern2, axis=1)
        ######################

        # 株価が欠損のレコードもあるので
        #  method='ffill'を使って、欠損している個所にもっとも近い個所にある有効なデーターで埋める
        stocks[code] = {'unit': unit,
                        'prices': prices.reindex(tse_index, method='ffill')}
    return stocks


def simulate_nikkei_tsumitate(db_file_name, start_date, end_date, deposit, reserve, code=1321):
    """
    初期所持金deposit円、毎月の初めにreserve円を入金し、その時の所持金で買えるだけ銘柄コード:codeの株を購入
    購入した銘柄はシミュレーション最終日まで売らずにホールドし、最終日に全部売却
    デフォルトでは日経平均に連動するETFのうち、最も出来高が大きい銘柄コード1321の「日経225連動型上場投資信託」を購入し続ける
    （rating_trade.pyの結果と比較するため）
    """
    # stocksは1銘柄の指定期間について、{code: {unit: xxx, price: [date, open, close]のデータフレーム}}
    stocks = create_stock_data(db_file_name, (code,), start_date, end_date)
    # print('stocks prices:', stocks[code]['prices'])

    def get_open_price_func(date, code):
        """指定日+指定銘柄の始値返す
        """
        # print(date, type(date))
        return stocks[code]['prices']['open'][date]

    def get_close_price_func(date, code):
        """指定日+指定銘柄の終値返す
        """
        return stocks[code]['prices']['close'][date]

    current_month = start_date.month - 1

    def trade_func(date, portfolio):
        """翌日の注文を決定する関数
        dateはループで回している日にち
        """
        # 関数の内側から外側の変数へのアクセスは基本的に「参照」のみが可能
        # 値を更新するには nonlocal 宣言をしなくてはならない
        nonlocal  current_month

        if date.month != current_month:
            portfolio.add_deposit(reserve)  # reserve分のお金を毎月加算していく
            # current_month = date.month  # 月初めの購入フラグはコメントアウトしていつでも金あれば買うようにする

            # 追加の株価購入条件もつける
            _date = datetime.date(date.year, date.month, date.day)
            if stocks[code]['prices']['buy_flag'][_date] == 1:

                return [sim.BuyMarketOrderAsPossible(code, stocks[code]['unit'])]  # 始値で買えるだけ買う

        return []

    return sim.simulate(start_date, end_date,
                        deposit,
                        trade_func,
                        get_open_price_func,
                        get_close_price_func)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-sd", "--start_date", type=str, default='20100401',
                    help="start day (yyyymmdd).")
    ap.add_argument("-ed", "--end_date", type=str, default='20200401',
                    help="end day (yyyymmdd).")
    ap.add_argument("-dep", "--deposit", type=int, default=1000000,
                    help="deposit money.")
    ap.add_argument("-res", "--reserve", type=int, default=50000,
                    help="reserve month money.")
    ap.add_argument("-c", "--code", type=int, default=1344,
                    help="brand code.")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()

    db_file_name = args['db_file_name']

    # シミュレーション期間
    start_date = datetime.date(int(args['start_date'][:4]),
                               int(args['start_date'][4:6]),
                               int(args['start_date'][6:]))
    end_date = datetime.date(int(args['end_date'][:4]),
                             int(args['end_date'][4:6]),
                             int(args['end_date'][6:]))

    # 開始時の所持金
    deposit = args['deposit']

    # 毎月増やす種銭。デフォルトは毎月5万円ずつ種銭を増やす
    reserve = args['reserve']

    # デフォルトの1144はTOPIX Core30に連動するETF:MAXIS トピックス･コア30上場投信
    code = args['code']

    # 1銘柄について、購入後持ち続けて最終日に売るシミュレーション実行
    portfolio, result = simulate_nikkei_tsumitate(db_file_name,
                                                  start_date, end_date,
                                                  deposit,
                                                  reserve,
                                                  code=code)
    print()
    print('現在の預り金:', portfolio.deposit)
    print('投資総額:', portfolio.amount_of_investment)
    print('総利益（税引き前）:', portfolio.total_profit)
    print('（源泉徴収)税金合計:', portfolio.total_tax)
    print('手数料合計:', portfolio.total_fee)
    print('保有銘柄 銘柄コード:', portfolio.stocks)
    # 日々の収益率を求める
    returns = (result.profit - result.profit.shift(1)) / result.price.shift(1)
    # 評価指標
    print('勝率:', round(portfolio.calc_winning_percentage(), 3), '%')  # 勝ちトレード回数/全トレード回数
    print('ペイオフレシオ:', round(portfolio.calc_payoff_ratio(), 5))  # 損益率: 勝ちトレードの平均利益額/負けトレードの平均損失額
    print('プロフィットファクター:', round(portfolio.calc_profit_factor(), 5))  # 総利益/総損失
    print('最大ドローダウン:', round(sim.calc_max_drawdown(result['price']) * 100, 3), '%')  # 累計利益または総資産額の最大落ち込み%。50%という値が出た場合、その戦略は使えない
    # リスクを考慮した評価指標
    # ※実装した指標の関数は、異なる売買戦略のシミュレーション結果を比較するためだけに利用すること
    # 　証券会社のサイトにもシャープレシオなどは載っているが、計算方法の前提が違う
    # 　計算方法がを比べても意味がない
    print('シャープレシオ:', round(sim.calc_sharp_ratio(returns), 5))  # リスク(株価のばらつき)に対するリターンの大きさ。0.5～0.9で普通、1.0～1.9で優秀、2.0以上だと大変優秀
    # print('インフォメーションレシオ:', sim.calc_information_ratio(returns, benchmark_retruns))  # 「リスクに対し、ベンチマークに対する超過リターンがどの程度か」を示す値
    print('ソルティノレシオ:', round(sim.calc_sortino_ratio(returns), 5))  # シャープレシオだけでは分からない下落リスクの抑制度合い
    print('カルマーレシオ:', round(sim.calc_calmar_ratio(returns, returns), 5))  # 同じ安全さに対してどれだけ利益を上げられそうかという指標。運用実績がより良いことを示す
    print()

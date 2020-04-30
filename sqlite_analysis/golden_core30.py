#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「買いシグナルであるゴールデンクロスで株を買い、売りシグナルであるデッドクロスで株を売る」のをsimulator.pyでテストする
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter3/golden_core30.py

＜シミュレーションの条件＞
・ある日に現在保有していない銘柄でゴールデンクロスが発生していたら、次の日にその銘柄を10万円以上となる最低の単元数だけ始値で買う。
　ただし所持金が足らない場合は買わない。
　なお、複数の銘柄で同時にゴールデンクロスが発生している場合、銘柄コードが小さいものを優先して買う。
・ある日に現在保有している銘柄でデッドクロスが発生していたら、次の日にその銘柄を全数、初値で売る。

Usage:
    $ activate stock
    $ python golden_core30.py
    $ python golden_core30.py -sd 20170101
"""
import sqlite3
import datetime
from collections import defaultdict
import pandas as pd

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim
import argparse


def create_stock_data(db_file_name, code_list, start_date, end_date):
    """指定した銘柄(code_list)それぞれの単元株数と日足(始値・終値）を含む辞書を作成
    """
    stocks = {}
    tse_index = sim.tse_date_range(start_date, end_date)
    conn = sqlite3.connect(db_file_name)
    for code in code_list:
        unit = conn.execute('SELECT unit from brands WHERE code = ?',
                            (code,)).fetchone()[0]
        prices = pd.read_sql('SELECT date, open, close '
                             'FROM prices '
                             'WHERE code = ? AND date BETWEEN ? AND ?'
                             'ORDER BY date',
                             conn,
                             params=(code, start_date, end_date),
                             parse_dates=('date',),
                             index_col='date')
        # print(prices)
        # 株価が欠損のレコードもあるので
        #  method='ffill'を使って、欠損している個所にもっとも近い個所にある有効なデーターで埋める
        stocks[code] = {'unit': unit,
                        'prices': prices.reindex(tse_index, method='ffill')}
    return stocks


def generate_cross_date_list(prices):
    """指定した日足データよりゴールデンクロス・デッドクロスが生じた日のリストを生成
    """
    # 移動平均を求める
    sma_5 = prices.rolling(window=5).mean()
    sma_25 = prices.rolling(window=25).mean()

    # 5日移動平均線と25日移動平均線のゴールデンクロス・デッドクロスが発生した場所を得る
    sma_5_over_25 = sma_5 > sma_25
    # shiftはデーターを指定した数だけずらすメソッド
    # shift(1)したデーターとしていないデーターを比較することで、
    # sma_5_over_25の値が変化した日、すなわちふたつの移動平均線がクロスした日のみTrueが格納されたSeriesであるcrossが求められます
    cross = sma_5_over_25 != sma_5_over_25.shift(1)
    golden_cross = cross & (sma_5_over_25 == True)
    dead_cross = cross & (sma_5_over_25 == False)
    # golden_crossとdead_crossの先頭25日分のデーターには正しくないデーターが格納されているため先頭25日分データを削除
    golden_cross.drop(golden_cross.head(25).index, inplace=True)
    dead_cross.drop(dead_cross.head(25).index, inplace=True)

    # 日付のリストに変換
    golden_list = [x.date()
                   for x
                   in golden_cross[golden_cross].index.to_pydatetime()]
    dead_list = [x.date()
                 for x
                 in dead_cross[dead_cross].index.to_pydatetime()]
    return golden_list, dead_list


def simulate_golden_dead_cross(db_file_name,
                               start_date, end_date,
                               code_list,
                               deposit,
                               order_under_limit):
    """ゴールデンクロス買い、売りシグナルであるデッドクロス売るシミュレーションコード
       deposit: 初期の所持金
    　 order_order_under_limit: ゴールデンクロス時の最小購入金額
    """
    # 指定した銘柄(code_list)それぞれの単元株数と日足(始値・終値）を含む辞書を作成
    stocks = create_stock_data(db_file_name, code_list, start_date, end_date)

    # {ゴールデンクロス・デッドクロスが発生した日 : 発生した銘柄のリスト}の辞書を作成
    # defaultdict の引数は「関数」です。関数を書けば、いかなる初期化でも可能
    golden_dict = defaultdict(list)
    dead_dict = defaultdict(list)
    for code in code_list:
        prices = stocks[code]['prices']['close']
        golden, dead = generate_cross_date_list(prices)
        for l, d in zip((golden, dead), (golden_dict, dead_dict)):
            for date in l:
                d[date].append(code)

    def get_open_price_func(date, code):
        """指定日+指定銘柄の始値返す
        """
        return stocks[code]['prices']['open'][date]

    def get_close_price_func(date, code):
        """指定日+指定銘柄の終値返す
        """
        return stocks[code]['prices']['close'][date]

    def trade_func(date, portfolio):
        """翌日の注文を決定する関数
        golden_dict/dead_dictを参照しながら買い注文・売り注文を行っていく
        """
        order_list = []
        # Dead crossが発生していて持っている株があれば売る
        if date in dead_dict:
            for code in dead_dict[date]:
                if code in portfolio.stocks:
                    order_list.append(
                        sim.SellMarketOrder(code,
                                            portfolio.stocks[code].current_count))
        # 保有していない株でgolden crossが発生していたら最低の単元数だけ買う
        if date in golden_dict:
            for code in golden_dict[date]:
                if code not in portfolio.stocks:
                    order_list.append(
                        sim.BuyMarketOrderMoreThan(code,
                                                   stocks[code]['unit'],
                                                   order_under_limit))
        return order_list

    return sim.simulate(start_date, end_date, deposit,
                        trade_func,
                        get_open_price_func, get_close_price_func)


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
    ap.add_argument("-cs", "--codes", type=int, nargs='*',
                    default=[2914, 3382, 4063, 4452, 4502, 4503, 4568, 6098, 6501, 6758,
                             6861, 6954, 6981, 7203, 7267, 7751, 7974, 8031, 8058, 8306,
                             8316, 8411, 8766, 8802, 9020, 9022, 9432, 9433, 9437, 9984],
                    help="brand codes.")
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

    # デフォルトはTOPIX Core30
    code_list = args['codes']

    # 開始時の所持金
    deposit = args['deposit']

    # ゴールデンクロス時の最小購入金額
    order_under_limit = deposit // 10

    # ゴールデンクロス買い、売りシグナルであるデッドクロス売るシミュレーション実行
    portfolio, result = simulate_golden_dead_cross(db_file_name,
                                                   start_date, end_date,
                                                   code_list,
                                                   deposit,
                                                   order_under_limit)
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

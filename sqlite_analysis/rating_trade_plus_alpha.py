#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毎月入金しながら目標株価をみて株を売買する戦略をsimulator.pyでテストする
https://raw.githubusercontent.com/BOSUKE/stock_and_python_book/master/chapter3/rating_trade.py

＜シミュレーションの条件＞
・ある日にn万円以上の所持金を持っていたら、新規に購入する株を物色する。
・購入する株は、物色日の1か月前から物色日までに公開された各証券会社の目標株価の平均値が物色日の終値と比べ20%より高い銘柄のうち、
　その割合が最も高いひとつの銘柄とする。
    ─1か月の間に、ある証券会社がある銘柄について目標株価を複数回発表している場合、最後に発表された目標株価を利用する。
    ─すでに保有している銘柄も購入対象とする。
・選んだ銘柄を、物色日の翌日に所持金で買えるだけ初値で買う。
　ただし、購入株数は単元株の倍数とし、また所持金が不足しており1単元も選んだ株を買えない時は購入しない。
・購入した銘柄の終値が平均取得価額の±20%になった場合、その株を翌日の初値ですべて売る。
・シミュレーション期間は、2008年4月1日から2018年4月1日の10年間。
・シミュレーション開始時の所持金は100万円。ただし毎月の月初めにn万円所持金を増やす。

Usage:
    $ activate stock
    $ python rating_trade_plus_alpha.py
    $ python rating_trade_plus_alpha.py -sd 20170101
"""
import argparse
import datetime
import sqlite3
import pandas as pd
from dateutil.relativedelta import relativedelta

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim


def buy_pattern2(row):
    """
    買い条件:
    - 当日終値が5MA,25MAより上
    - 当日終値が前日からの直近10日間の最大高値より上
    - 当日出来高が前日比50%以上増
    - 当日終値が当日の値幅の上位70%以上(大陽線?)
    - 当日終値が25MA乖離率+5％未満
    - 翌日始値が前日終値より10円高い
    """
    if row['close'] >= row['5MA'] \
        and row['close'] >= row['25MA'] \
        and row['close'] >= row['high_shfi1_10MAX'] \
        and row['volume_1diff_rate'] >= 0.5 \
        and row['close'] >= ((row['high'] - row['low']) * 0.7) + row['low'] \
        and (row['close'] - row['25MA']) / row['25MA'] < 0.05 \
        and row['open_close_1diff'] > 10:
        # return row
        return 1
    else:
        # return pd.Series()
        return 0


def is_judge_stock_code(db_file_name, code, start_date, end_date):
    """指定した銘柄(code)が追加条件を満たすか判定
    """
    conn = sqlite3.connect(db_file_name)
    prices = pd.read_sql('SELECT * '
                         'FROM prices '
                         'WHERE code = ? AND date BETWEEN ? AND ?'
                         'ORDER BY date',
                         conn,
                         params=(code, start_date, end_date),
                         parse_dates=('date',),
                         index_col='date')
    if prices.empty == False:
        # ### plu_alpha #### #
        prices['5MA'] = prices['close'].rolling(window=5).mean()
        prices['25MA'] = prices['close'].rolling(window=25).mean()
        prices['75MA'] = prices['close'].rolling(window=75).mean()
        prices['200MA'] = prices['close'].rolling(window=200).mean()
        prices['close_10MAX'] = prices['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
        prices['high_10MAX'] = prices['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
        prices['high_shfi1_10MAX'] = prices['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
        prices['volume_1diff_rate'] = (prices['volume'] - prices['volume'].shift(1).fillna(0)) / prices['volume'].shift(1).fillna(0)  # 前日比出来高
        prices['open_close_1diff'] = prices['open'].shift(-1).fillna(0) - prices['close']  # 翌日始値-当日終値
        # 購入フラグ付ける
        prices['buy_flag'] = prices.apply(buy_pattern2, axis=1)
        ######################
        prices = prices.dropna(how='any')

        # 条件に当てはまるレコード無ければ、行もしくは列がない
        if prices.empty:
            return False
        print(prices)
        buy_flag_date = [x.strftime("%Y-%m-%d") for x in prices.index.tolist()][-1]
        # print(code, buy_flag_date)
        if end_date == buy_flag_date:
            return True
    return False


def simulate_rating_trade(db_file_name, start_date, end_date, deposit, reserve,
                          months=1,
                          rating_rate=1.2, sell_buy_rate=0.2, minimum_buy_threshold=300000):
    conn = sqlite3.connect(db_file_name)

    def get_open_price_func(date, code):
        """DBから指定の銘柄の指定の日の始値を取得する
        """
        r = conn.execute('SELECT open FROM prices '
                         'WHERE code = ? AND date <= ? '
                         'ORDER BY date DESC LIMIT 1',
                         (code, date)).fetchone()
        return r[0]

    def get_close_price_func(date, code):
        """DBから指定の銘柄の指定の日の終値を取得する
        """
        r = conn.execute('SELECT close FROM prices '
                         'WHERE code = ? AND date <= ? '
                         'ORDER BY date DESC LIMIT 1',
                         (code, date)).fetchone()
        return r[0]

    def get_prospective_brand(date):
        """購入する銘柄を物色  購入すべき銘柄の(コード, 単元株数, 比率)を返す
        →「物色日の1か月前から物色日までに公開された各証券会社の
        　目標株価の平均値が物色日の終値と比べ20%より高い銘柄のうち、
        　その割合が最も高いひとつの銘柄
        　（ただし、1か月の間にある証券会社がある銘柄について目標株価を複数回発表している場合、
        　最後に発表された目標株価を利用する）」
        を見つけ出して、その銘柄の銘柄コードと単元株数を返す
        """
        prev_month_day = date - relativedelta(months=months)
        # <SQL解説>
        # WITH句の last_date_t で銘柄コードと証券会社ごとに、1か月前から物色日までの間で最後に目標株価を公開した日を求める
        # ↓
        # WITH句の avg_t で銘柄ごとに、それぞれの証券会社が1か月前から物色日までの間で最後に公表した目標株価の平均値を求める
        # ↓
        # 残りの SELECT で銘柄ごとに物色日の終値に対する求めた平均値の割合が20%より高い銘柄のうち、
        # 最も割合が高い1銘柄の銘柄コードと単元株数（と割合）を求める。
        sql = f"""
        WITH last_date_t AS (
            SELECT
                MAX(date) AS max_date,
                code,
                think_tank
            FROM
                ratings
            WHERE
                date BETWEEN :prev_month_day AND :day
            GROUP BY
                code,
                think_tank
        ),  avg_t AS (
            SELECT
                ratings.code,
                AVG(ratings.target) AS target_avg
            FROM
                ratings,
                last_date_t
            WHERE
                ratings.date = last_date_t.max_date
                AND ratings.code = last_date_t.code
                AND ratings.think_tank = last_date_t.think_tank
            GROUP BY
                ratings.code
        )
        SELECT
            avg_t.code,
            brands.unit,
            (avg_t.target_avg / raw_prices.close) AS rate
        FROM
            avg_t,
            raw_prices,
            brands
        WHERE
            avg_t.code = raw_prices.code
            AND raw_prices.date = :day
            AND rate > {rating_rate}
            AND raw_prices.code = brands.code
        ORDER BY
            rate DESC
        """
        return conn.execute(sql,
                            {'day': date,
                             'prev_month_day': prev_month_day}).fetchall()

    current_month = start_date.month - 1

    def trade_func(date, portfolio):
        """注文を決定する関数
        """
        nonlocal current_month
        if date.month != current_month:
            # 月初め => 入金
            portfolio.add_deposit(reserve)
            current_month = date.month

        order_list = []

        # ±20パーセントで利確/損切り
        for code, stock in portfolio.stocks.items():
            current = get_close_price_func(date, code)
            rate = (current / stock.average_cost) - 1
            if abs(rate) > sell_buy_rate:
                order_list.append(
                    sim.SellMarketOrder(code, stock.current_count))

        # 月の入金額以上持っていたら新しい株を物色
        if portfolio.deposit >= reserve:
            # レーティング情報
            for r in get_prospective_brand(date):
                if r:
                    code, unit, _ = r
                    # order_list.append(sim.BuyMarketOrderAsPossible(code, unit))

                    # print(type(start_date), type(date))
                    # レーティングよくてチャートもいいのは基本無い
                    _start_date = date - datetime.timedelta(days=500)  # 200日移動平均線の計算があるから500日前からスタート
                    print('code, start_date, date:', code, start_date, date,
                                                     is_judge_stock_code(db_file_name, code, _start_date, date))

                    # 追加の株価購入条件もつける
                    if is_judge_stock_code(db_file_name, code, start_date, date):
                        # 購入最低金額以上持っていたら新しい株購入
                        order_list.append(sim.BuyMarketOrderMoreThan(code,
                                                                     unit,
                                                                     minimum_buy_threshold))

        return order_list

    return sim.simulate(start_date, end_date, deposit,
                        trade_func,
                        get_open_price_func, get_close_price_func)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-sd", "--start_date", type=str, default='20150401',
                    help="start day (yyyymmdd).")
    ap.add_argument("-ed", "--end_date", type=str, default='20200401',
                    help="end day (yyyymmdd).")
    ap.add_argument("-dep", "--deposit", type=int, default=1000000,
                    help="deposit money.")
    ap.add_argument("-res", "--reserve", type=int, default=0,
                    help="reserve month money.")
    ap.add_argument("-r_rate", "--rating_rate", type=float, default=1.2,
                    help="target rating rate.")
    ap.add_argument("-sb_rate", "--sell_buy_rate", type=float, default=0.2,
                    help="sell buy price rate.")
    ap.add_argument("-m", "--months", type=int, default=1,
                    help="rating months. default=3")
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000,
                    help="minimum_buy_threshold.")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_args()
    # print(str(args))

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

    # 証券会社のレーティング情報の株価 >= 現在株価 * rating_rate の銘柄を買い
    rating_rate = args['rating_rate']

    # 購入時の株価 * (1+sell_buy_rate) 以上（利確）もしくは購入時の株価 * (1-sell_buy_rate) 以下（損切り）
    sell_buy_rate = args['sell_buy_rate']

    # 毎月入金しながら目標株価をみて株を売買するシミュレーション実行
    portfolio, result = simulate_rating_trade(db_file_name,
                                              start_date, end_date,
                                              deposit, reserve,
                                              months=args['months'],
                                              rating_rate=rating_rate,
                                              sell_buy_rate=sell_buy_rate,
                                              minimum_buy_threshold=args['minimum_buy_threshold'])
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
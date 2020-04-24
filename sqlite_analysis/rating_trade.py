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
    $ python rating_trade.py
"""
import argparse
import datetime
import sqlite3
from dateutil.relativedelta import relativedelta

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim


def simulate_rating_trade(db_file_name, start_date, end_date, deposit, reserve,
                          rating_rate=1.2, sell_buy_rate=0.2):
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
        prev_month_day = date - relativedelta(months=1)
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
        LIMIT
            1
        """
        return conn.execute(sql,
                            {'day': date,
                             'prev_month_day': prev_month_day}).fetchone()

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
            r = get_prospective_brand(date)
            if r:
                code, unit, _ = r
                order_list.append(sim.BuyMarketOrderAsPossible(code, unit))

        return order_list

    return sim.simulate(start_date, end_date, deposit,
                        trade_func,
                        get_open_price_func, get_close_price_func)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-sd", "--start_date", type=str, default='20100401',
                    help="start day (yyyy/mm/dd).")
    ap.add_argument("-ed", "--end_date", type=str, default='20200401',
                    help="end day (yyyy/mm/dd).")
    ap.add_argument("-dep", "--deposit", type=int, default=1000000,
                    help="deposit money.")
    ap.add_argument("-res", "--reserve", type=int, default=50000,
                    help="reserve month money.")
    ap.add_argument("-r_rate", "--rating_rate", type=float, default=1.2,
                    help="target rating rate.")
    ap.add_argument("-sb_rate", "--sell_buy_rate", type=float, default=0.2,
                    help="sell buy price rate.")
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

    # 証券会社のレーティング情報の株価 >= 現在株価 * rating_rate の銘柄を買い
    rating_rate = args['rating_rate']

    # 購入時の株価 * (1+sell_buy_rate) 以上（利確）もしくは購入時の株価 * (1-sell_buy_rate) 以下（損切り）
    sell_buy_rate = args['sell_buy_rate']

    # 毎月入金しながら目標株価をみて株を売買するシミュレーション実行
    portfolio, result = simulate_rating_trade(db_file_name,
                                              start_date, end_date,
                                              deposit, reserve,
                                              rating_rate=rating_rate,
                                              sell_buy_rate=sell_buy_rate)
    print()
    print('現在の預り金:', portfolio.deposit)
    print('投資総額:', portfolio.amount_of_investment)
    print('総利益（税引き前）:', portfolio.total_profit)
    print('（源泉徴収)税金合計:', portfolio.total_tax)
    print('手数料合計:', portfolio.total_fee)
    print('保有銘柄 銘柄コード:', portfolio.stocks)
    print()

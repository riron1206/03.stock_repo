#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1銘柄について、開始日に購入後持ち続けて最終日に売るのをsimulator.pyでテストする
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter3/buy_and_hold.py

＜シミュレーションの条件＞
・開始日に成行注文で買えるだけ買う（始値で買う）。
・最終日にその銘柄を全数始値で売る。

Usage:
    $ activate stock
    $ python buy_and_hold.py
"""
import argparse
import datetime

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim
from golden_core30 import create_stock_data


def simulate_buy_and_hold(db_file_name,
                          start_date, end_date,
                          deposit,
                          code=1344):
    """1銘柄(code)について、start_dateに購入後持ち続けて最終日(end_date)に売るシミュレーションコード
    デフォルトの銘柄(1344)はTOPIX Core30に連動するETF:MAXIS トピックス･コア30上場投信
    （golden_core30.pyの結果と比較するため）
    """
    # 指定した銘柄(code_list)それぞれの単元株数と日足(始値・終値）を含む辞書を作成
    stocks = create_stock_data(db_file_name, (code,), start_date, end_date)

    # 株価データないときのエラー避けるために全てnanのレコードは削除する
    # print(stocks[code]['prices'].shape)
    # print(stocks[code]['prices'].dropna(how='all').shape)
    stocks[code]['prices'] = stocks[code]['prices'].dropna(how='all')

    def get_open_price_func(date, code):
        """指定日+指定銘柄の始値返す
        """
        ## return stocks[code]['prices'].iloc[0]['open']  # 対応するdateで株価データないときのエラー避けるために0番目のレコードから取る
        return stocks[code]['prices']['open'][date]

    def get_close_price_func(date, code):
        """指定日+指定銘柄の終値返す
        """
        ## return stocks[code]['prices'].iloc[-1]['close']  # 対応するdateで株価データないときのエラー避けるために無理やり最後のレコードから取る場合
        return stocks[code]['prices']['close'][date]

    def trade_func(date, portfolio):
        """注文を決定する関数
        """
        if date == start_date:
            return [sim.BuyMarketOrderAsPossible(code, stocks[code]['unit'])]
        return []

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

    # デフォルトの1144はTOPIX Core30に連動するETF:MAXIS トピックス･コア30上場投信
    code = args['code']

    # 1銘柄について、購入後持ち続けて最終日に売るシミュレーション実行
    portfolio, result = simulate_buy_and_hold(db_file_name,
                                              start_date, end_date,
                                              deposit,
                                              code)
    print()
    print('現在の預り金:', portfolio.deposit)
    print('投資総額:', portfolio.amount_of_investment)
    print('総利益（税引き前）:', portfolio.total_profit)
    print('（源泉徴収)税金合計:', portfolio.total_tax)
    print('手数料合計:', portfolio.total_fee)
    print('保有銘柄 銘柄コード:', portfolio.stocks)
    print()

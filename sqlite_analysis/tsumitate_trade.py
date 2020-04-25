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
    $ python buy_and_hold.py
"""
import datetime
import argparse

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import simulator as sim
from golden_core30 import create_stock_data


def simulate_nikkei_tsumitate(db_file_name, start_date, end_date, deposit, reserve, code=1321):
    """
    初期所持金deposit円、毎月の初めにreserve円を入金し、その時の所持金で買えるだけ銘柄コード:codeの株を購入
    購入した銘柄はシミュレーション最終日まで売らずにホールドし、最終日に全部売却
    デフォルトでは日経平均に連動するETFのうち、最も出来高が大きい銘柄コード1321の「日経225連動型上場投資信託」を購入し続ける
    （rating_trade.pyの結果と比較するため）
    """
    stocks = create_stock_data(db_file_name, (code,), start_date, end_date)

    def get_open_price_func(date, code):
        return stocks[code]['prices']['open'][date]

    def get_close_price_func(date, code):
        return stocks[code]['prices']['close'][date]

    current_month = start_date.month - 1

    def trade_func(date, portfolio):
        # 関数の内側から外側の変数へのアクセスは基本的に「参照」のみが可能
        # 値を更新するには nonlocal 宣言をしなくてはならない
        nonlocal  current_month

        if date.month != current_month:
            portfolio.add_deposit(reserve)
            current_month = date.month
            return [sim.BuyMarketOrderAsPossible(code, stocks[code]['unit'])]
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
                    help="start day (yyyy/mm/dd).")
    ap.add_argument("-ed", "--end_date", type=str, default='20200401',
                    help="end day (yyyy/mm/dd).")
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
    print()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    $ activate stock
    $ python original_trade.py
"""
import argparse
import datetime
import sqlite3
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

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
        return 1
    else:
        return None


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
    # ### plu_alpha #### #
    prices['5MA'] = prices['close'].rolling(window=5).mean()
    prices['25MA'] = prices['close'].rolling(window=25).mean()
    prices['75MA'] = prices['close'].rolling(window=75).mean()
    prices['200MA'] = prices['close'].rolling(window=200).mean()
    prices['close_10MAX'] = prices['close'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大終値
    prices['high_10MAX'] = prices['high'].rolling(window=10, min_periods=0).max()  # 直近10日間の中で最大高値
    prices['high_shfi1_10MAX'] = prices['high'].shift(1).fillna(0).rolling(window=10, min_periods=0).max()  # 前日からの直近10日間の中で最大高値
    prices['high_shfi1_15MAX'] = prices['high'].shift(1).fillna(0).rolling(window=15, min_periods=0).max()  # 前日からの直近15日間の中で最大高値
    prices['volume_1diff_rate'] = (prices['volume'] - prices['volume'].shift(1).fillna(0)) / prices['volume']  # 前日比出来高
    # 購入フラグ付ける
    prices['buy_flag'] = prices.apply(buy_pattern2, axis=1)
    ######################
    prices = prices.dropna(how='any')

    # 条件に当てはまるレコード無ければ、行もしくは列がない
    if prices.shape[0] == 0 or prices.shape[1] == 0:
        return False
    # print(prices.head())
    buy_flag_date = [x.strftime("%Y-%m-%d") for x in prices.index.tolist()][-1]
    # print(code, end_date, buy_flag_date, type(end_date), type(buy_flag_date))
    # print(end_date == buy_flag_date)
    if end_date == datetime.datetime.strptime(buy_flag_date, '%Y-%m-%d').date():
        return True


def simulate_rating_trade(db_file_name, start_date, end_date, deposit, reserve,
                          months=1,
                          rating_rate=1.1, sell_buy_rate=0.2, minimum_buy_threshold=300000):
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

    def get_brand_volume_df(date):
        # codes = '2914, 7013'
        codes = '7013, 8304, 3407, 2502, 2802, 4503, 6857, 6113, 6770, 8267, 7202, 5019, 8001, 4208, 4523, 5201, 9202, 6472, 9613, 9437, 6361, 8725, 2413, 6103, 9532, 3861, 4578, 1802, 6703, 9007, 6645, 7733, 4452, 6952, 1812, 9107, 7012, 9503, 2801, 7751, 6971, 4151, 2503, 6326, 3405, 8253, 9008, 9009, 9433, 5406, 1605, 9766, 4902, 6301, 1721, 7186, 4751, 2501, 3436, 6674, 5411, 5020, 6473, 3086, 4507, 8355, 4911, 7762, 1803, 9104, 4004, 4063, 8303, 5401, 9412, 7735, 7269, 7270, 5232, 4005, 5713, 6302, 8053, 5802, 8830, 6724, 1928, 9735, 4689, 3382, 2768, 6758, 8729, 9984, 8630, 4568, 8750, 6367, 1801, 7912, 4506, 5541, 5233, 6976, 1925, 8601, 8233, 2531, 4502, 8331, 4519, 9502, 8795, 2432, 3401, 6762, 4631, 4543, 4061, 6902, 4324, 5301, 9022, 3289, 8035, 8766, 9531, 9005, 8804, 9501, 4042, 5332, 9001, 9602, 5707, 5901, 3101, 3402, 5714, 4043, 7911, 7203, 8015, 4704, 7731, 9021, 2871, 1963, 4021, 7201, 2002, 3105, 6988, 5202, 5333, 4272, 5703, 1332, 6471, 3863, 5631, 4041, 2914, 9062, 6701, 5214, 9432, 2282, 6178, 9101, 8604, 1808, 6752, 7832, 9020, 6305, 6501, 7004, 7205, 9983, 6954, 8028, 8354, 5803, 6702, 6504, 4901, 5108, 5801, 7267, 8628, 7261, 8252, 1333, 8002, 8411, 7003, 4183, 5706, 8309, 8316, 8031, 8801, 3099, 4188, 7211, 7011, 8058, 9301, 8802, 6503, 5711, 8306, 6479, 2269, 6506, 9064, 7951, 7272, 3103, 6841, 5101, 6098, 7752, 8308'
        sql = f"""
        SELECT
            B.code,
            B.unit,
            P.volume
        FROM
            prices AS P,
            brands AS B
        WHERE
            P.date = :date
            AND P.code = B.code
            AND P.code IN ({codes})
        ORDER BY
            volume DESC
        """
        return pd.read_sql(sql, conn,
                           params={'date': date})

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

        # 購入最低金額以上持っていたら新しい株を物色
        if portfolio.deposit >= minimum_buy_threshold:
            # 当日の出来高が高い銘柄順に並び替えたデータフレーム
            df = get_brand_volume_df(date)
            #pbar = tqdm(df.head(50).itertuples(index=False))
            #for row in pbar:
            #    pbar.set_description(str(date))
            for row in df.head(50).itertuples(index=False):

                # 追加の株価購入条件もつける
                if is_judge_stock_code(db_file_name, row.code, start_date, date):
                    # order_list.append(sim.BuyMarketOrderAsPossible(code, unit))

                    # print(row.code, date)
                    # 購入最低金額以上持っていたら新しい株購入
                    order_list.append(sim.BuyMarketOrderMoreThan(row.code,
                                                                 row.unit,
                                                                 minimum_buy_threshold))

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
    ap.add_argument("-ed", "--end_date", type=str, default=None,
                    help="end day (yyyymmdd).")
    ap.add_argument("-dep", "--deposit", type=int, default=1000000,
                    help="deposit money.")
    ap.add_argument("-res", "--reserve", type=int, default=0,
                    help="reserve month money.")
    ap.add_argument("-r_rate", "--rating_rate", type=float, default=1.1,
                    help="target rating rate.")
    ap.add_argument("-sb_rate", "--sell_buy_rate", type=float, default=0.2,
                    help="sell buy price rate.")
    ap.add_argument("-m", "--months", type=int, default=3,
                    help="rating months. default=3")
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=100000,
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
    if args['end_date'] is None:
        end_date = datetime.date.today()
    else:
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

    # 購入時の最低価格。デフォルトは30万円。30万以上持ち金あれば、30万円ぐらいの量株買う
    minimum_buy_threshold = args['minimum_buy_threshold']

    # レーティング情報サーチする期間
    months = args['months']

    # 毎月入金しながら目標株価をみて株を売買するシミュレーション実行
    portfolio, result = simulate_rating_trade(db_file_name,
                                              start_date, end_date,
                                              deposit, reserve,
                                              months=months,
                                              rating_rate=rating_rate,
                                              sell_buy_rate=sell_buy_rate,
                                              minimum_buy_threshold=minimum_buy_threshold)
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
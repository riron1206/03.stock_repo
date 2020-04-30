#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
営業利益が拡大している銘柄を買ったときをsimulator.pyでテストする
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter4_5/opincome_trade.py

＜シミュレーションの条件＞
1.過去3期の営業利益前年同期比が10%以上かつ期ごとに増加している企業を選ぶ
・2017年4～6月期に対する2018年4～6月期の営業利益の増加率をX1とし、
・2017年1～3月期に対する2018年1～3月期の営業利益の増加率をX2とし、
・2016年10～12月期に対する2017年10～12月期の営業利益の増加率をX3
としたときに、
・X1 > X2 > X3 > 10％を満たす銘柄

2.出来高が1億円以上の企業を選ぶ
- その日に四半期報告書を公開し、かつその日の出来高が1億円以上である銘柄から、1の条件を満たすものを選出

3.購入銘柄の最終決定と購入
- 1と2の条件をともに満たす銘柄が複数ある場合は、3期分の営業利益の前年同期比の平均が最も高いものを、次の日の購入銘柄として選択
- 購入株数は代金が30万円以上となる最低の単元数
- 所持金が不足してる場合は買わない

4.利確・損切りのタイミングは2パターン用意
- 購入した銘柄の終値が平均取得価額の＋10％となった場合、その銘柄を次の日にすべて売る
- 終値が平均取得価額より下回ったケースにおいては、損切りのタイミングの違いがどのような影響を及ぼすかの確認のために、
　-10％と-5％で損切りする2つのパターンでシミュレーション

Usage:
    $ activate stock
    $ python opincome_trade.py
"""
import datetime
import argparse
import sqlite3
import pandas as pd
import numpy as np
import simulator as sim


def simulate_op_income_trade(db_file_name,
                             start_date,
                             end_date,
                             deposit,
                             growth_rate_threshold,
                             minimum_buy_threshold,
                             trading_value_threshold,
                             profit_taking_threshold,
                             stop_loss_threshold):
    """
    営業利益が拡大している銘柄を買う戦略のシミュレーション
    Args:
        db_file_name: DB(SQLite)のファイル名
        start_date:   シミュレーション開始日
        end_date:     シミュレーション終了日
        deposit:      シミュレーション開始時点での所持金
        growth_rate_threshold : 購入対象とする銘柄の四半期成長率の閾値
        minimum_buy_threshold : 購入時の最低価格
        trading_value_threshold: 購入対象とする銘柄の出来高閾値
        profit_taking_threshold: 利確を行う閾値
        stop_loss_threshold:   : 損切りを行う閾値
    """
    conn = sqlite3.connect(db_file_name)

    def get_open_price_func(date, code):
        """date日におけるcodeの銘柄の初値の取得"""
        r = conn.execute('SELECT open FROM prices '
                         'WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 1',
                         (code, date)).fetchone()
        return r[0]

    def get_close_price_func(date, code):
        """date日におけるcodeの銘柄の終値の取得"""
        r = conn.execute('SELECT close FROM prices '
                         'WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT 1',
                         (code, date)).fetchone()
        return r[0]

    def get_op_income_df(date):
        """date日の出来高が閾値以上であるに四半期決算を公開した銘柄の銘柄コード、
        単元株、date日以前の四半期営業利益の情報、date日以前の四半期決算公表日を
        取得する
        """
        return pd.read_sql("""
                            WITH target AS (
                                SELECT
                                    code,
                                    unit,
                                    term
                                FROM
                                    quarterly_results
                                    JOIN prices
                                    USING(code, date)
                                    JOIN brands
                                    USING(code)
                                WHERE
                                    date = :date
                                    AND close * volume > :threshold
                                    AND op_income IS NOT NULL
                            )
                            SELECT
                                code,
                                unit,
                                op_income,
                                results.term as term
                            FROM
                                target
                                JOIN quarterly_results AS results
                                USING(code)
                            WHERE
                                results.term <= target.term
                            """,
                           conn,
                           params={"date": date,
                                   "threshold": trading_value_threshold})

    def check_income_increasing(income):
        """3期分の利益の前年同期比が閾値以上かつ単調増加であるかを判断。
        閾値以上単調増加である場合は3期分の前年同期比の平均値を返す。
        条件を満たさない場合は0を返す
        """
        if len(income) < 7 or any(income <= 0):
            return 0

        t1 = (income.iat[0] - income.iat[4]) / income.iat[4]
        t2 = (income.iat[1] - income.iat[5]) / income.iat[5]
        t3 = (income.iat[2] - income.iat[6]) / income.iat[6]
        # print('t1, t2, t3', t1, t2, t3)
        if (t1 > t2) and (t2 > t3) and (t3 > growth_rate_threshold):
            return np.average((t1, t2, t3))
        else:
            return 0

    def choose_best_stock_to_buy(date):
        """date日の購入対象銘柄の銘柄情報・単位株を返す"""
        df = get_op_income_df(date)
        # print(df.head(3))
        found_code = None
        found_unit = None
        max_rate = 0
        for code, f in df.groupby("code"):
            income = f.sort_values("term", ascending=False)[:7]
            # print('code', code)
            # print('income', income)
            rate = check_income_increasing(income["op_income"])
            if rate > max_rate:
                max_rate = rate
                found_code = code
                found_unit = income["unit"].iat[0]

        return found_code, found_unit

    def trade_func(date, portfolio):
        """date日の次の営業日の売買内容を決定する関数"""
        order_list = []

        # 売却する銘柄の決定
        for code, stock in portfolio.stocks.items():
            current = get_close_price_func(date, code)
            rate = (current / stock.average_cost) - 1
            # print('rate', rate)
            if rate >= profit_taking_threshold or rate <= -stop_loss_threshold:
                order_list.append(
                    sim.SellMarketOrder(code, stock.current_count))

        # 購入する銘柄の決定
        code, unit, = choose_best_stock_to_buy(date)
        if code:
            order_list.append(sim.BuyMarketOrderMoreThan(code,
                                                         unit,
                                                         minimum_buy_threshold))

        return order_list

    # シミュレータの呼び出し
    return sim.simulate(start_date, end_date, deposit,
                        trade_func, get_open_price_func, get_close_price_func)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-sd", "--start_date", type=str, default='20181001',
                    help="start day (yyyymmdd).")
    ap.add_argument("-ed", "--end_date", type=str, default='20200401',
                    help="end day (yyyymmdd).")
    ap.add_argument("-dep", "--deposit", type=int, default=3000000,
                    help="deposit money.")
    ap.add_argument("-grt", "--growth_rate_threshold", type=float, default=0.10,
                    help="growth_rate_threshold. default=+10%")
    ap.add_argument("-mbt", "--minimum_buy_threshold", type=int, default=300000,
                    help="minimum_buy_threshold.. default=300,000 yen")
    ap.add_argument("-tvt", "--trading_value_threshold", type=int, default=100,
                    help="trading_value_threshold. default=100 million yen")
    ap.add_argument("-ptt", "--profit_taking_threshold", type=float, default=0.10,
                    help="profit_taking_threshold. default=+10%")
    ap.add_argument("-slt", "--stop_loss_threshold", type=float, default=0.10,
                    help="stop_loss_threshold. default=-10%")
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

    # 購入対象とする銘柄の四半期成長率の閾値。デフォルトは10%
    growth_rate_threshold = args['growth_rate_threshold']

    # 購入時の最低価格。デフォルトは30万円。30万以上持ち金あれば、30万円ぐらいの量株買う
    minimum_buy_threshold = args['minimum_buy_threshold']

    # 購入対象とする銘柄の出来高閾値。デフォルトは1億
    trading_value_threshold = args['trading_value_threshold']

    # 利確を行う閾値。デフォルトは10%上がったら売り
    profit_taking_threshold = args['profit_taking_threshold']

    # 損切りを行う閾値。デフォルトは10%下がったら売り
    stop_loss_threshold = args['stop_loss_threshold']

    # 営業利益が拡大している銘柄を買う戦略のシミュレーション実行
    portfolio, result = simulate_op_income_trade(db_file_name,
                                                 start_date,
                                                 end_date,
                                                 deposit,
                                                 growth_rate_threshold,
                                                 minimum_buy_threshold,
                                                 trading_value_threshold,
                                                 profit_taking_threshold,
                                                 stop_loss_threshold)
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

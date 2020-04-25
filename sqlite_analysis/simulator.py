#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株売買シュミレーションを行うためのコード
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter3/simulator.py
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter4_5/simulator.py

＜シミュレーターの要件＞
・シミュレーターは、指定された日付の範囲内の東証が開いている日について、1日単位で株の売買をシミュレートする。
・テスト対象である取引戦略は、ある日のザラバ終了後にその日までの情報をもとに次の日に売買する株を決定する。
・売買は始値で約定したとみなす。
    ─買うことを決めていた株の翌日の始値＋手数料が所持金を上回る場合、株の購入は行わない。
    ─売買に必要なコストを支払うことができる状態であれば、始値で約定したものとみなす。（自分の注文が約定しないケースを考慮しない）
・売買のたびに手数料を差し引く。（楽天証券の手数料体系を適用する）
・源泉徴収ありの特定口座を想定し、売却のたびに利益の累計に応じた税金を源泉徴収する。
    ─売却の結果、利益が減った場合はそれまでに源泉徴収していた税金を還元する。
・シミュレーターは、1日のシミュレートごとに資産の評価額と利益の計算を行う。
    ─評価額の算出はその日の終値を用いて行う。
・価格がつかなかった日の始値・終値は、その前日の始値・終値を用いる。（前日も価格が付かなかったのであれば、さらにその前日のデーターを使う）
・シミュレーション最終日に、保有しているすべての株を売却する。

＜シミュレーターの処理内容＞
日付内の東証が開いている日ごとにブロック内の処理を実施（最終日除く) {
    * 前日に行われた注文を執行
        - 購入ならば始値*株数+手数料を所持金から差し引いて、保有株数を増やす
        - 売却ならば始値*株数-手数料-税金を所持金に加えて、保有株数を減らす
    * ★ 本日までの情報をもとに明日実行する注文を決定 ★
    * 本日の終値を用いて評価額と利益を算出
}
最終日に実施 {
    * 保有する株価をすべて始値で売却
    * 本日の終値を用いて評価額と利益を算出
}
"""
import math
import collections
import pandas as pd

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/Git/japandas/')
from japandas import japandas


def calc_tax(total_profit):
    """儲けに対する税金計算
    """
    if total_profit < 0:
        return 0
    return int(total_profit * 0.20315)


def calc_fee(total):
    """約定手数料計算(楽天証券の場合）
    """
    if total <= 50000:
        return 54
    elif total <= 100000:
        return 97
    elif total <= 200000:
        return 113
    elif total <= 500000:
        return 270
    elif total <= 1000000:
        return 525
    elif total <= 1500000:
        return 628
    elif total <= 30000000:
        return 994
    else:
        return 1050


def calc_cost_of_buying(count, price):
    """株を買うのに必要なコストと手数料を計算
    """
    subtotal = int(count * price)
    fee = calc_fee(subtotal)
    return subtotal + fee, fee


def calc_cost_of_selling(count, price):
    """株を売るのに必要なコストと手数料を計算
    """
    subtotal = int(count * price)
    fee = calc_fee(subtotal)
    return fee, fee


def calc_max_drawdown(prices):
    """最大ドローダウンを計算して返す
        Usege:
            # calc_max_drawdownの呼び出し
            calc_max_drawdown(result.price)
    """
    cummax_ret = prices.cummax()  # cummax関数: DataFrameまたはSeries軸の累積最大値を求める。あるindexについて、そのindexとそれより前にある全要素の最大値を求めて、その結果をそのindexに格納したSeries（またはDataFrame）を返すメソッド
    drawdown = cummax_ret - prices  # 日々の総資産額のその日までの最大値とその日の総資産額との差分
    max_drawdown_date = drawdown.idxmax()  # drawdownの中で最大の値をもつ要素のindexをidxmaxメソッドで求め、そのindexを使って最大ドローダウンの値を求めている
    return drawdown[max_drawdown_date] / cummax_ret[max_drawdown_date]


def calc_sharp_ratio(returns):
    """シャープレシオを計算して返す
    """
    # .meanは平均値(=期待値)を求めるメソッド
    return returns.mean() / returns.std()


def calc_information_ratio(returns, benchmark_retruns):
    """インフォメーションレシオを計算して返す
    """
    excess_returns = returns - benchmark_retruns
    return excess_returns.mean() / excess_returns.std()


def calc_sortino_ratio(returns):
    """ソルティノレシオを計算して返す
    """
    tdd = math.sqrt(returns.clip_upper(0).pow(2).sum() / returns.size)
    return returns.mean() / tdd


def calc_sortino_bench(returns, benchmark_retruns):
    excess_returns = returns - benchmark_retruns
    return calc_sortino_ratio(excess_returns)


def calc_calmar_ratio(prices, returns):
    """カルマ―レシオを計算して返す
    """
    return returns.mean() / calc_max_drawdown(prices)


class OwnedStock(object):
    def __init__(self):
        self.total_cost = 0     # 取得にかかったコスト（総額)
        self.total_count = 0    # 取得した株数(総数)
        self.current_count = 0  # 現在保有している株数
        self.average_cost = 0   # 平均取得価額

    def append(self, count, cost):
        """保有する株数が増えるたびに平均取得価額を計算する
        複数回にわけて株を購入した場合、売却時点で取得にかかったコストを見積もるため平均取得価額計算する
        """
        if self.total_count != self.current_count:
            self.total_count = self.current_count
            self.total_cost = self.current_count * self.average_cost
        self.total_cost += cost
        self.total_count += count
        self.current_count += count
        self.average_cost = math.ceil(self.total_cost / self.total_count)

    def remove(self, count):
        if self.current_count < count:
            raise ValueError("can't remove", self.total_cost, count)
        self.current_count -= count


class Portfolio(object):
    """資産管理を行うクラス
    今、どの株をどれだけ持っているか、その評価額、株の売買にかかるコスト（手数料・税金）などを保持する
    """
    def __init__(self, deposit):
        self.deposit = deposit  # 現在の預り金
        self.amount_of_investment = deposit  # 投資総額
        self.total_profit = 0  # 総利益（税引き前）
        self.total_tax = 0  # （源泉徴収)税金合計
        self.total_fee = 0  # 手数料合計
        self.count_of_trades = 0  # トレード総数
        self.count_of_wins = 0   # 勝ちトレード数
        self.total_gains = 0     # 総利益(損失分の相殺無しの値)
        self.total_losses = 0    # 総損出

        self.stocks = collections.defaultdict(OwnedStock)  # 保有銘柄 銘柄コード　-> OwnedStock への辞書

    def add_deposit(self, deposit):
        """預り金を増やす (= 証券会社に入金)
        """
        self.deposit += deposit
        self.amount_of_investment += deposit

    def buy_stock(self, code, count, price):
        """株を買う
        """
        cost, fee = calc_cost_of_buying(count, price)
        if cost > self.deposit:
            raise ValueError('cost > deposit', cost, self.deposit)

        # 保有株数増加
        self.stocks[code].append(count, cost)

        self.deposit -= cost
        self.total_fee += fee

    def sell_stock(self, code, count, price):
        """株を売る
        """
        subtotal = int(count * price)
        cost, fee = calc_cost_of_selling(count, price)
        if cost > self.deposit + subtotal:
            raise ValueError('cost > deposit + subtotal',
                             cost, self.deposit + subtotal)

        # 保有株数減算
        stock = self.stocks[code]
        average_cost = stock.average_cost
        stock.remove(count)
        if stock.current_count == 0:
            del self.stocks[code]

        # 儲け計算
        profit = int((price - average_cost) * count - cost)
        self.total_profit += profit

        # トレード結果保存
        self.count_of_trades += 1
        if profit >= 0:
            self.count_of_wins += 1
            self.total_gains += profit
        else:
            self.total_losses += -profit

        # 源泉徴収額決定
        current_tax = calc_tax(self.total_profit)
        withholding = current_tax - self.total_tax
        self.total_tax = current_tax

        self.deposit += subtotal - cost - withholding
        self.total_fee += fee

    def calc_current_total_price(self, get_current_price_func):
        """現在の評価額を返す
        """
        stock_price = sum(get_current_price_func(code)
                          * stock.current_count
                          for code, stock in self.stocks.items())
        return stock_price + self.deposit

    def calc_winning_percentage(self):
        """勝率を返す"""
        return (self.count_of_wins / self.count_of_trades) * 100

    def calc_payoff_ratio(self):
        """ペイオフレシオを返す
        """
        loss = self.count_of_trades - self.count_of_wins
        if self.count_of_wins and loss:
            ave_gain = self.total_gains / self.count_of_wins
            ave_losses = self.total_losses / loss
            return ave_gain / ave_losses
        else:
            return sys.float_info.max

    def calc_profit_factor(self):
        """プロフィットファクターを返す
        """
        if self.total_losses:
            return self.total_gains / self.total_losses
        else:
            return sys.float_info.max


class Order(object):
    """注文はOrderクラスとして表現
    翌日に実行したい注文のパターンは様々なバリエーションが考えられるため、
    Orderクラスをスーパークラスとして注文のバリエーションごとにサブクラスを作成
    翌日に実行したい注文のパターンごとにexecuteメソッドをオーバーライドしたOrderサブクラスを作成して利用する
    """
    def __init__(self, code):
        self.code = code

    def execute(self, date, portfolio, get_price_func):
        pass

    @classmethod
    def default_order_logger(cls, order_type, date, code, count, price, before_deposit, after_deposit):
        print("{} {} code:{} count:{} price:{} deposit:{} -> {}".format(
            date.strftime('%Y-%m-%d'),
            order_type,
            code,
            count,
            price,
            before_deposit,
            after_deposit
        ))
    logger = default_order_logger


class BuyMarketOrderAsPossible(Order):
    """残高で買えるだけ買う成行注文
    """

    def __init__(self, code, unit):
        super().__init__(code)
        self.unit = unit

    def execute(self, date, portfolio, get_price_func):
        price = get_price_func(self.code)
        # print(portfolio.deposit, price, self.unit)
        count_of_buying_unit = int(portfolio.deposit / price / self.unit)
        while count_of_buying_unit:
            try:
                count = count_of_buying_unit * self.unit
                prev_deposit = portfolio.deposit
                portfolio.buy_stock(self.code, count, price)
                self.logger("BUY", date, self.code, count, price, prev_deposit, portfolio.deposit)
            except ValueError:
                count_of_buying_unit -= 1
            else:
                break


class BuyMarketOrderMoreThan(Order):
    """指定額以上で最小の株数を買う
    """
    def __init__(self, code, unit, under_limit):
        super().__init__(code)
        self.unit = unit
        self.under_limit = under_limit

    def execute(self, date, portfolio, get_price_func):
        price = get_price_func(self.code)
        unit_price = price * self.unit
        if unit_price > self.under_limit:
            count_of_buying_unit = 1
        else:
            count_of_buying_unit = int(self.under_limit / unit_price)
        while count_of_buying_unit:
            try:
                count = count_of_buying_unit * self.unit
                prev_deposit = portfolio.deposit
                portfolio.buy_stock(self.code, count, price)
                self.logger("BUY", date, self.code, count, price, prev_deposit, portfolio.deposit)
            except ValueError:
                count_of_buying_unit -= 1
            else:
                break


class SellMarketOrder(Order):
    """成行の売り注文
    """
    def __init__(self, code, count):
        super().__init__(code)
        self.count = count

    def execute(self, date, portfolio, get_price_func):
        price = get_price_func(self.code)
        prev_deposit = portfolio.deposit
        portfolio.sell_stock(self.code, self.count, price)
        self.logger("SELL", date, self.code, self.count, price, prev_deposit, portfolio.deposit)


def tse_date_range(start_date, end_date):
    """
    指定した範囲内の東証（TSE）の営業日をpandasのDatetimeIndexの形式（簡単に言えば日付の配列的なもの）で返す関数
    東証の休業日の情報取るのにjapandasを使う
    """
    tse_business_day = pd.offsets.CustomBusinessDay(
        calendar=japandas.TSEHolidayCalendar())
    return pd.date_range(start_date, end_date,
                         freq=tse_business_day)


def simulate(start_date: int, end_date: int, deposit: int,
             trade_func,
             get_open_price_func,
             get_close_price_func):
    """
    [start_date, end_date]の範囲内の売買シミュレーションを行う
    deposit: 最初の所持金
    trade_func:
        シミュレーションする取引関数
        （引数 date, portfolio でOrderのリストを返す関数）
    get_open_price_func:
        指定銘柄コードの指定日の始値を返す関数 (引数 date, code)
    get_close_price_func:
        指定銘柄コードの指定日の終値を返す関数 (引数 date, code)
    """

    portfolio = Portfolio(deposit)

    total_price_list = []
    profit_or_loss_list = []

    def record(d):
        # 本日(d)の損益などを記録
        current_total_price = portfolio.calc_current_total_price(
            lambda code: get_close_price_func(d, code))
        total_price_list.append(current_total_price)
        profit_or_loss_list.append(current_total_price
                                   - portfolio.amount_of_investment)

    def execute_order(d, orders):
        # 本日(d)において注文(orders)をすべて執行する
        for order in orders:
            order.execute(d, portfolio,
                          lambda code: get_open_price_func(d, code))

    order_list = []
    date_range = [pdate.to_pydatetime().date()
                  for pdate in tse_date_range(start_date, end_date)]
    for date in date_range[:-1]:
        execute_order(date, order_list)           # 前日に行われた注文を執行
        order_list = trade_func(date, portfolio)  # 明日実行する注文を決定する
        record(date)                              # 損益等の記録

    # 最終日に保有株は全部売却
    last_date = date_range[-1]
    execute_order(last_date,
                  [SellMarketOrder(code, stock.current_count)
                   for code, stock in portfolio.stocks.items()])
    record(last_date)

    return portfolio, \
           pd.DataFrame(data={'price': total_price_list,
                              'profit': profit_or_loss_list},
                        index=pd.DatetimeIndex(date_range))

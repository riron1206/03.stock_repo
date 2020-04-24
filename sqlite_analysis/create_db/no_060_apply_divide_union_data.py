#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調整後株価のSQLiteへの格納と更新
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter2/apply_divide_union_data.py
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「prices」「applied_divide_union_data」テーブル作成
    CREATE TABLE prices (
        code TEXT, -- 銘柄コード
        date TEXT, -- 日付
        open REAL, -- 初値
        high REAL, -- 高値
        low REAL, -- 安値
        close REAL, -- 終値
        volume INTEGER, -- 出来高
        PRIMARY KEY(code, date)
    );
    CREATE TABLE applied_divide_union_data (
        code TEXT, -- 銘柄コード
        date_of_right_allotment TEXT, -- 権利確定日
        PRIMARY KEY(code, date_of_right_allotment)
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_060_apply_divide_union_data.py
"""
import argparse
import datetime
import sqlite3
from tqdm import tqdm


def copy_prices_ratings_table(db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """
    株価分割併合による株価更新処理失敗したとき用
    prices, ratings テーブルを raw_* から作り直す
    ※間違ったデータ入れてしまいcsvからテーブル再作成するときに使った
    Usage:
        import no_060_apply_divide_union_data
        no_060_apply_divide_union_data.copy_prices_ratings_table()
    """
    with sqlite3.connect(db_file_name) as conn:
        conn.cursor().execute("INSERT INTO prices(code, date, open, high, low, close, volume) SELECT code, date, open, high, low, close, volume FROM raw_prices")
    with sqlite3.connect(db_file_name) as conn:
        conn.cursor().execute("INSERT INTO ratings(date, code, think_tank, rating, target) SELECT date, code, think_tank, rating, target FROM raw_ratings")


def apply_divide_union_data(db_file_name, date_of_right_allotment,
                            update_tables=['prices', 'ratings', 'applied_divide_union_data']):
    """
    分割・併合が行われるたびにdivide_union_dataテーブルをもとに prices, ratings, applied_divide_union_data テーブルを更新する
    """
    conn = sqlite3.connect(db_file_name)

    # date_of_right_allotment(デフォルトでは今日) 以前の
    # 分割・併合データで未適用のもの(divide_union_data テーブルにはあるが, applied_divide_union_data テーブルにはないレコード）を取得する
    sql = """
    SELECT
        d.code, d.date_of_right_allotment, d.before, d.after
    FROM
        divide_union_data AS d
    WHERE
        d.date_of_right_allotment < ?
        AND NOT EXISTS (
            SELECT
                *
            FROM
                applied_divide_union_data AS a
            WHERE
                d.code = a.code
                AND d.date_of_right_allotment = a.date_of_right_allotment
            )
    ORDER BY
        d.date_of_right_allotment
    """
    cur = conn.execute(sql, (date_of_right_allotment,))
    divide_union_data = cur.fetchall()
    # print('divide_union_data:', divide_union_data)

    with conn:
        conn.execute('BEGIN TRANSACTION')
        for code, date_of_right_allotment, before, after in tqdm(divide_union_data):

            # 例えば、DeNA （銘柄コード: 2432 ）は 2010 年 5 月 26 日に755000円だった株価が次の日の5月27日には2661円になっている
            # これは1株を300株とする株式分割を行ったため
            # 1株の価値を現在の株価と合わせるために株価分割併合起きた以前の株価を分割/併合数で掛け算して調整する

            # before個あった株をafter個に分割/併合する
            # 上記のDeNAの場合、before=300, after=1 なので掛け算する値であるrate = 1/300 になる
            rate = after / before
            inv_rate = 1 / rate  # 出来高も逆数で掛け算必要

            # prices 更新
            # ※大量件数のテーブルUPDATEなので時間かかる
            conn.execute(
                f'UPDATE {update_tables[0]} SET '
                ' open = open * :rate, '
                ' high = high * :rate, '
                ' low = low  * :rate, '
                ' close = close * :rate, '
                ' volume = volume * :inv_rate '
                'WHERE code = :code '
                ' AND date < :date_of_right_allotment',
                {'code': code,
                 'date_of_right_allotment': date_of_right_allotment,
                 'rate': rate,
                 'inv_rate': inv_rate})

            # ratings 更新
            conn.execute(
                f'UPDATE {update_tables[1]} SET '
                ' target = target * :rate '
                'WHERE code = :code '
                ' AND date < :date_of_right_allotment',
                {'code': code,
                 'date_of_right_allotment': date_of_right_allotment,
                 'rate': rate})

            try:
                # applied_divide_union_data 更新
                conn.execute(
                    f'INSERT INTO {update_tables[2]}(code, date_of_right_allotment) VALUES(?,?)',
                    (code, date_of_right_allotment))
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    # 今日またはそれ以前が権利確定日である未適用の分割・併合データーがあれば更新する
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    apply_divide_union_data(db_file_name, today)
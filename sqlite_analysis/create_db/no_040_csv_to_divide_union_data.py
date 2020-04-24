#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日足CSVファイルから分割・併合情報を求めてSQLiteに格納
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter2/csv_to_divide_union_data.py
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「divide_union_data」テーブル作成
    CREATE TABLE divide_union_data (
        code TEXT, -- 銘柄コード
        date_of_right_allotment TEXT, -- 権利確定日
        before REAL, -- 分割・併合前株数の割合
        after REAL, -- 分割・併合後株数の割合
        PRIMARY KEY("code","date_of_right_allotment")
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_040_csv_to_divide_union_data.py
"""
import argparse
import csv
import glob
import datetime
import os
import sqlite3
from tqdm import tqdm


def generater_devide_union_from_csv_file(csv_file_name, code):
    """
    調整前終値と調整後終値のデータから分割・併合がいつ、どのような割合で行われたか求める
    調整前終値と調整後終値のデータは、株式投資メモやYahoo!ファイナンスからダウンロードしたデータならついている
    """
    with open(csv_file_name, encoding="shift_jis") as f:
        reader = csv.reader(f)
        next(reader)  # 先頭行を飛ばす

        def parse_recode(row):
            if '-' in row[0]:
                d = datetime.datetime.strptime(row[0], '%Y-%m-%d').date()  # 日付
            else:
                d = datetime.datetime.strptime(row[0], '%Y/%m/%d').date()  # 日付
            r = float(row[4])  # 調整前終値
            a = float(row[6])  # 調整後終値
            return d, r, a

        _, r_n, a_n = parse_recode(next(reader))
        for row in reader:
            d, r, a = parse_recode(row)

            # 調整前終値/調整後終値が0の時、ゼロ除算でエラーになるため、いずれkが0の場合は計算しないことにする
            if int(a) != 0 and int(a_n) != 0 and int(r) != 0 and int(r_n) != 0:

                # 連続する2日間の調整前終値の変化率と調整後終値の変化率 = (調整後終値 * 分割・併合の権利確定日の調整前終値) / (調整後終値 * 翌営業日の権利落ち日の調整前終値)
                rate = (a_n * r) / (a * r_n)

                if abs(rate - 1) > 0.005:
                    if rate < 1:
                        before = round(1 / rate, 2)
                        after = 1
                    else:
                        before = 1
                        after = round(rate, 2)
                    yield code, d, before, after
            r_n = r
            a_n = a


def all_csv_file_to_divide_union_table(db_file_name, csv_file_dir, csv_name="*.T.csv", table_name='divide_union_data'):
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        # 1csvごとに処理
        pbar = tqdm(glob.glob(os.path.join(csv_file_dir, csv_name)))
        for path in pbar:
            pbar.set_description(path)
            file_name = os.path.basename(path)
            code = file_name.split('.')[0]
            # csvの株価とcsvのファイル名からもらった銘柄コードを使ってinsert
            for record in generater_devide_union_from_csv_file(path, code):
                try:
                    c.execute(f'INSERT INTO {table_name} (code,date_of_right_allotment,before,after) VALUES(?,?,?,?)', record)
                except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-dir", "--csv_file_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabuoji3',
                    help="stock csv file dir path.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    csv_file_dir = args['csv_file_dir']

    all_csv_file_to_divide_union_table(db_file_name, csv_file_dir, csv_name="*.csv")

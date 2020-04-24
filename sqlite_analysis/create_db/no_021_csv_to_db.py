#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo!ファイナンスのCSVファイルの内容をSQLiteに格納
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter2/csv_to_db.py
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「raw_prices」テーブル作成
    ※DB Browser for SQLiteは作成したテーブルの自動保存されない。テーブル作ったらDB Browser for SQLite閉じて保存すること
    CREATE TABLE raw_prices (
        code TEXT, -- 銘柄コード
        date TEXT, -- 日付
        open REAL, -- 初値
        high REAL, -- 高値
        low REAL, -- 安値
        close REAL, -- 終値
        volume INTEGER, -- 出来高
        PRIMARY KEY(code, date)
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_021_csv_to_db.py
"""
import argparse
import csv
import glob
import datetime
import os
import sqlite3
import time
import traceback
from tqdm import tqdm
import pandas as pd


def insert_recent_data_and_save_csv(db_file_name, code, table_names=['raw_prices', 'prices'], csv_file=None):
    """
    株式投資メモから最新の時系列データを引っ張ってinsertする
    csvファイルのパスを渡す場合はそのcsvに新規レコード追加する
    ※スクレイピングなのでサーバに負荷掛かる。やりすぎないこと
    """
    # time.sleep(2)  # 1銘柄のスクレイピングに2秒間隔開ける。サーバに負荷かけて逮捕されないために
    try:
        df = pd.read_html(f'https://kabuoji3.com/stock/{code}/')[0]

        if csv_file is not None:
            csv_df = pd.read_csv(csv_file, encoding='shift-jis')
            csv_last_date = csv_df.loc[csv_df.shape[0] - 1]['日付']  # 2020-04-13形式の文字列
            csv_last_date = datetime.datetime(int(csv_last_date[:4]),
                                              int(csv_last_date[5:7]),
                                              int(csv_last_date[8:10]))  # csvファイルの最終日

            # スクレイピングしたデータフレームをcsvファイルの最終日以降のレコードだけにする
            df['日付_datetime'] = pd.to_datetime(df['日付'])
            df = df[df['日付_datetime'] > csv_last_date].reset_index(drop=True)
            df = df.sort_values(by=['日付_datetime'], ascending=True)
            df = df.drop(['日付_datetime'], axis=1)

            # csvファイルに追加レコード入れて保存
            csv_df = pd.concat([csv_df, df], ignore_index=True)
            csv_df.to_csv(csv_file, index=False, encoding="SHIFT-JIS")

        df = df.rename(columns={'日付': 'date', '始値': 'open', '高値': 'high', '安値': 'low', '終値': 'close', '出来高': 'volume'})
        df['code'] = code
        df = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
        with sqlite3.connect(db_file_name) as conn:
            c1 = conn.cursor()
            c2 = conn.cursor()
            for row in tqdm(df.itertuples(index=False)):
                try:
                    # insert raw_prices
                    c1.execute(f'INSERT INTO {table_names[0]} (code,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)',
                               (row.code, row.date, row.open, row.high, row.low, row.close, row.volume))
                except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                    pass
                try:
                    # insert prices
                    c2.execute(f'INSERT INTO {table_names[1]} (code,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)',
                               (row.code, row.date, row.open, row.high, row.low, row.close, row.volume))
                except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                    pass
        return df  # 一応データフレーム返す
    except Exception as e:
        traceback.print_exc()
        return str(code) + ': ' + str(e)


def generate_price_from_csv_file(csv_file_name, code):
    with open(csv_file_name, encoding="shift_jis") as f:
        reader = csv.reader(f)
        next(reader)  # 先頭行を飛ばす
        for row in reader:
            if '-' in row[0]:
                d = datetime.datetime.strptime(row[0], '%Y-%m-%d').date()  # 日付
            else:
                d = datetime.datetime.strptime(row[0], '%Y/%m/%d').date()  # 日付
            o = float(row[1])  # 初値
            h = float(row[2])  # 高値
            l = float(row[3])  # 安値
            c = float(row[4])  # 終値
            v = int(row[5])    # 出来高
            yield code, d, o, h, l, c, v


def generate_from_csv_dir(csv_dir, generate_func, csv_name="*.csv"):
    pbar = tqdm(glob.glob(os.path.join(csv_dir, csv_name)))
    for path in pbar:
        pbar.set_description(path)
        file_name = os.path.basename(path)
        code = file_name.split('.')[0]
        for d in generate_func(path, code):
            yield d


def all_csv_file_to_db(db_file_name, csv_file_dir):
    price_generator = generate_from_csv_dir(csv_file_dir, generate_price_from_csv_file)
    conn = sqlite3.connect(db_file_name)
    with conn:
        sql = """
        INSERT INTO raw_prices(code,date,open,high,low,close,volume)
        VALUES(?,?,?,?,?,?,?)
        """
        conn.executemany(sql, price_generator)

        # raw_pricesテーブルをpriceテーブルに複製する
        c = conn.cursor()
        c.execute("INSERT INTO prices(code, date, open, high, low, close, volume) SELECT code, date, open, high, low, close, volume FROM raw_prices")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-dir", "--csv_file_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabuoji3',
                    help="stock csv file dir path.")
    ap.add_argument("-u", "--is_update", action='store_const', const=True, default=False,
                    help="table update flag.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']
    csv_file_dir = args['csv_file_dir']

    if args['is_update']:
        # csvファイル名から銘柄コード取得する
        pbar = tqdm(glob.glob(os.path.join(csv_file_dir, "*.csv")))
        for path in pbar:
            pbar.set_description(path)
            code = os.path.basename(path).split('.')[0]
            _ = insert_recent_data_and_save_csv(db_file_name, code, csv_file=path)
    else:
        all_csv_file_to_db(db_file_name, csv_file_dir)

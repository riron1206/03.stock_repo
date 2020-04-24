#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQueryで上場・廃止情報スクレイピングしてSQLiteへinsert
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter2/get_new_brands.py
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「new_brands」「delete_brands」テーブル作成
    ※DB Browser for SQLiteは作成したテーブルの自動保存されない。テーブル作ったらDB Browser for SQLite閉じて保存すること
    CREATE TABLE new_brands ( -- 上場情報
        code TEXT, -- 銘柄コード
        date TEXT, -- 上場日
        PRIMARY KEY(code, date)
    );
    CREATE TABLE delete_brands ( -- 廃止情報
        code TEXT, -- 銘柄コード
        date TEXT, -- 廃止日
        PRIMARY KEY(code, date)
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_030_get_new_delete_brands_info.py
"""
from tqdm import tqdm
from pyquery import PyQuery
import datetime
import sqlite3
import argparse

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
import no_010_get_brands_info


def insert_new_brands(db_file_name, select_table_name='new_brands', insert_table_name='brands'):
    """
    insert_new_delete_brands_to_db()で追加した上場銘柄の情報をbrandsデーブルにinsertする
    廃止銘柄は削除せずそのままにしておく
    """
    conn = sqlite3.connect(db_file_name)
    with conn:
        code_range = []
        c = conn.cursor()
        # select new_brands
        for row in c.execute(f'SELECT * FROM {select_table_name}'):
            code_range.append(row[0])
    # insert brands
    no_010_get_brands_info.insert_brands_to_db(db_file_name, sorted(code_range), table_name=insert_table_name)
    return code_range  # 一応insertする銘柄コード返しておく


def delete_brands_generator():
    """日本取引所グループのサイトから廃止銘柄スクレイピングして、generatorで渡す
    """
    url = 'https://www.jpx.co.jp/listing/stocks/delisted/index.html'
    q = PyQuery(url)
    for d, i in tqdm(zip(q.find('tbody > tr:even > td:eq(0)'), q.find('tbody > tr:even > td:eq(2)'))):
        date = datetime.datetime.strptime(d.text, '%Y/%m/%d').date()
        yield (i.text, date)


def new_brands_generator():
    """日本取引所グループのサイトから上場銘柄スクレイピングして、generatorで渡す
    """
    url = 'http://www.jpx.co.jp/listing/stocks/new/index.html'
    q = PyQuery(url)
    for d, i in tqdm(zip(q.find('tbody > tr:even > td:eq(0)'), q.find('tbody > tr:even span'))):
        date = datetime.datetime.strptime(d.text, '%Y/%m/%d').date()
        yield (i.get('id'), date)


def insert_new_delete_brands_to_db(db_file_name, table_name, generate_func):
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        for code_date in generate_func:
            try:
                c.execute(f'INSERT INTO {table_name}(code,date) VALUES(?,?)', code_date)
                print(table_name, 'insert OK:', code_date)
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    # 上場情報insert
    insert_new_delete_brands_to_db(db_file_name, 'new_brands', new_brands_generator())
    # 廃止情報insert
    insert_new_delete_brands_to_db(db_file_name, 'delete_brands', delete_brands_generator())
    # 上場情報をbrandsテーブルに反映
    _ = insert_new_brands(db_file_name)

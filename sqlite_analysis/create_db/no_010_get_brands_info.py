#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
銘柄コードなどの株価以外の銘柄基本情報を「株探」からPyQueryでスクレイピングしてSQLiteへinsert
https://github.com/BOSUKE/stock_and_python_book/blob/master/chapter2/get_brands.py
Usage:
    1. DB Browser for SQLite(SQLiteの管理ツール)インストール
    インストーラー: http://sqlitebrowser.org/
    使い方: https://www.dbonline.jp/sqlite-db-browser/database/index1.html

    2. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「brands」テーブル作成
    CREATE TABLE brands (
        code TEXT PRIMARY KEY, -- 銘柄コード
        name TEXT, -- 銘柄名（正式名称）
        short_name TEXT, -- 銘柄名（略称）
        market TEXT, -- 上場市場名
        sector TEXT, -- セクタ
        unit INTEGER -- 単元株数
    );

    3. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    ※全銘柄(1301-9908)データ入手するのに約3時間かかる
    $ activate stock
    $ python no_010_get_brands_info.py  # 「brans」テーブルにデータ入る
"""
import argparse
from pyquery import PyQuery
import time
import sqlite3
from tqdm import tqdm


def get_brand(code: str):
    """ 銘柄コードから株価情報スクレイピング """
    url = 'https://kabutan.jp/stock/?code={}'.format(code)

    q = PyQuery(url)

    if len(q.find('div.company_block')) == 0:
        return None

    try:
        name = q.find('div.company_block > h3').text()
        code_short_name = q.find('#stockinfo_i1 > div.si_i1_1 > h2').text()
        short_name = code_short_name[code_short_name.find(" ") + 1:]
        market = q.find('span.market').text()
        unit_str = q.find('#kobetsu_left > table:nth-child(4) > tbody > tr:nth-child(6) > td').text()
        unit = int(unit_str.split()[0].replace(',', ''))
        sector = q.find('#stockinfo_i2 > div > a').text()
    except (ValueError, IndexError):
        return None

    return code, name, short_name, market, unit, sector


def brands_generator(code_range: list):
    """ generatorで指定された範囲の銘柄コードに対する銘柄情報を次々に生成 """
    for code in tqdm(code_range):
        brand = get_brand(code)
        if brand:
            yield brand
        time.sleep(1)


def insert_brands_to_db(db_file_name: str, code_range: list, table_name='brands'):
    """ SQLiteのbrandsテーブルにレコードinsert """
    # ---- 一気にcommitする ---- #
    # conn = sqlite3.connect(db_file_name)
    # with conn:
    #     c = conn.cursor()
    #     sql = f'INSERT INTO {table_name}(code,name,short_name,market,unit,sector) VALUES(?,?,?,?,?,?)'
    #     conn.executemany(sql, brands_generator(code_range))

    # ---- 1レコードづつcommitする ---- #
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        for code_date in brands_generator(code_range):
            try:
                c.execute(f'INSERT INTO {table_name}(code,name,short_name,market,unit,sector) VALUES(?,?,?,?,?,?)', code_date)
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    # 本書執筆時点で最も小さな有効な銘柄コードである1301から最も大きな9997の範囲で動かす
    insert_brands_to_db(db_file_name, range(1301, 9998))
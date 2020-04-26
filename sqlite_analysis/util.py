#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" util関数 """
import sqlite3
import pandas as pd
import pathlib
import time


def download_1stock_df(code, save_dir: str, start_year: int, end_year: int):
    """
    pandasで株式投資メモの株価を1銘柄ダウンロードしてDataFrameで返す
    ※スクレイピングなのでサーバに負荷掛かる。やりすぎないこと
    """
    # time.sleep(2)  # 1銘柄のスクレイピングに2秒間隔開ける。サーバに負荷かけて逮捕されないために
    df_concat = None
    for year in range(int(start_year), int(end_year) + 1):
        url = f'https://kabuoji3.com/stock/{code}/{year}/'
        try:
            df = pd.read_html(url)[0]
            if df_concat is None:
                df_concat = df
            else:
                df_concat = pd.concat([df_concat, df], ignore_index=True)
        except Exception as e:
            # print(code, year, 'error:', e)
            pass
    return df_concat


def execute_sql(sql, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """ sqlite3でSQL実行する """
    with sqlite3.connect(db_file_name) as conn:
        conn.cursor().execute(sql)


def check_table(table_name, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """
    sqlite3でテーブルの中身を確認する
    $ "sqlite3 TEST.db"
    > SELECT * FROM persons;
    と同じこと
    """
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        c.execute(f'SELECT * FROM {table_name}')
        # print(c.fetchall())  # 中身を全て取得するfetchall()を使って、printする。
        return c.fetchall()


def show_tables(db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """
    sqlite3でテーブル一覧取得
    https://crimnut.hateblo.jp/entry/2018/04/17/172709
    $ "sqlite3 TEST.db"
    > .tables #.tablesでDBのテーブル一覧を取得
    と同じこと
    """
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        c.execute("select * from sqlite_master where type='table'")
        # 確認
        # for x in c.fetchall():
        #     print(x)
        return c.fetchall()


def csv_to_table(csv_path, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db', table_columns=None, table_name=None):
    """
    csv内のデータをDataFrameとして読み出し、sqlite3でDBへ書き込む
    https://qiita.com/saira/items/e08c8849cea6c3b5eb0c
    """
    df = pd.read_csv(csv_path, encoding='shift-jis')
    if table_columns is not None:
        # カラム名（列ラベル）を作成。csv file内にcolumn名がある場合は、下記は不要 pandasが自動で1行目をカラム名として認識してくれる。
        df.columns = table_columns
    if table_name is None:
        table_name = str(pathlib.Path.stem(csv_path))
    with sqlite3.connect(db_file_name) as conn:
        # 読み込んだcsvファイルをsqlに書き込む
        # if_exists='append'でinsert
        # index=Falseでindexはinsertしないようにする
        df.to_sql(table_name, conn, if_exists='append', index=False)
        # 確認 作成したデータベースを1行ずつ見る
        # select_sql = f'SELECT * FROM {table_name}'
        # for row in conn.cursor().execute(select_sql):
        #     print(row)


def table_to_df(table_name=None, sql=None, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """ sqlite3で指定テーブルのデータをDataFrameで返す """
    conn = sqlite3.connect(db_file_name)
    if table_name is not None:
        return pd.read_sql(f'SELECT * FROM {table_name}', conn)
    elif sql is not None:
        return pd.read_sql(sql, conn)
    else:
        return None


def fetch_prices_to_df(code, db_file_name=r'D:\DB_Browser_for_SQLite\stock.db', table_name='prices'):
    """  指定銘柄コードについて、sqlite3でpricesテーブルSELECTしてDataFrameで返す """
    conn = sqlite3.connect(db_file_name)
    return pd.read_sql(f'SELECT date, open, high, low, close, volume FROM {table_name} WHERE code = ? ORDER BY date',
                       conn,
                       params=(code,),
                       parse_dates=('date',),
                       index_col='date')

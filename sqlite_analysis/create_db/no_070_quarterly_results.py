#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3ヵ月決算【実績】データをSQLiteへ格納と更新
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「quarterly_results」テーブル作成
    CREATE TABLE quarterly_results (
        code TEXT, -- 銘柄コード
        term TEXT, -- 決算期 （例: 2018年4～6月期ならば2018-06）
        date TEXT, -- 決算発表日
        sales INTEGER, -- 売上高（単位:百万円）
        op_income INTEGER, -- 営業利益（単位:百万円）
        ord_income INTEGER, -- 経常利益（単位:百万円）
        net_income INTEGER -- 最終利益（単位:百万円）
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_070_quarterly_results.py
"""
import argparse
import datetime
import glob
import os
import pathlib
import re
import sqlite3
import traceback
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def csv_to_quarterly_results_table(save_dir=r'D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly',
                                   db_file_name=r'D:\DB_Browser_for_SQLite\stock.db'):
    """
    csvファイルからDBに入れる
    ※間違ったデータ入れてしまいcsvからテーブル再作成するときに使った
    ※quarterly_results テーブル削除→中身空で再作成してから実行すること
    Usage:
        import no_070_quarterly_results
        no_070_quarterly_results.csv_to_quarterly_results_table()
    """
    pbar = tqdm(glob.glob(os.path.join(save_dir, "*.csv")))
    for path in pbar:
        pbar.set_description(path)
        code = os.path.basename(path).split('.')[0]
        csv_df = pd.read_csv(path, encoding='shift-jis')
        df = csv_df[['決算期', '発表日', '売上高', '営業益', '経常益', '最終益']]
        df = df.rename(columns={'決算期': 'term', '発表日': 'date', '売上高': 'sales',
                                '営業益': 'op_income', '経常益': 'ord_income', '最終益': 'net_income'})
        insert_quarterly_df_to_table(code, df, db_file_name)


def download_quarterly_csv(code, save_dir=None, csv_path=None):
    """
    株探（https://kabutan.jp/）から3ヵ月決算【実績】をスクレイピングして保存して、データフレームを返す
    業績の単位は「百万円」
    csvファイルのパスを渡す場合はそのcsvに新規レコード追加する
    ※https://kabutan.jp/robots.txt より株探の各銘柄の業績ページはスクレイピング禁止されていない（許可されているわけではない）が。やりすぎると逮捕されかねないので注意
    ※有報キャッチャー(https://ufocatch.com/Dcompany.aspx?ec=E22016&m=y)の方が情報リッチだがseleniumで取る必要がありそうなのでやめた
    """
    try:
        dfs = pd.read_html(f'https://kabutan.jp/stock/finance?code={code}')
        for df in dfs:
            if set(df.columns) == set(['決算期', '売上高', '営業益', '経常益', '最終益', '修正1株益', '売上営業損益率', '発表日']):
                break

        df['発表日'] = df['発表日'].replace('－', np.nan)  # 発表日が－の銘柄(1444など)があったので欠損にする
        df = df.dropna(subset=['発表日'])

        for col in ['売上高', '営業益', '経常益', '最終益']:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # DBに入れる業績列は数値のみにする
        df = df.dropna(how='any')

        if df.shape[0] > 0:
            for col in ['修正1株益', '売上営業損益率']:
                df[col] = df[col].replace('－', '-')  # －の文字はshft-jisではエラーになるから置換

            df['決算期'] = df['決算期'].map(lambda x: re.sub("\*$", '', x))
            df['決算期'] = '20' + df['決算期'].map(lambda x: x[-8:])
            df['決算期'] = df['決算期'].map(lambda x: re.sub('\.\d{2}-', '-', x))

            df['発表日'] = '20' + df['発表日']
            df['発表日'] = df['発表日'].str.replace('/', '-')
            # print(df, df.shape)

            if csv_path is None:
                # この段階でcsvに保存
                csv_path = os.path.join(save_dir, str(code) + '.csv')
                df.to_csv(csv_path, index=False, encoding="SHIFT-JIS")
            else:
                csv_df = pd.read_csv(csv_path, encoding='shift-jis')
                csv_last_date = csv_df.loc[csv_df.shape[0] - 1]['発表日']  # 2020-04-13形式の文字列
                csv_last_date = datetime.datetime(int(csv_last_date[:4]),
                                                  int(csv_last_date[5:7]),
                                                  int(csv_last_date[8:10]))  # csvファイルの最終日

                # スクレイピングしたデータフレームをcsvファイルの最終日以降のレコードだけにする
                df['発表日_datetime'] = pd.to_datetime(df['発表日'])
                df = df[df['発表日_datetime'] > csv_last_date].reset_index(drop=True)
                df = df.sort_values(by=['発表日_datetime'], ascending=True)
                df = df.drop(['発表日_datetime'], axis=1)

                # csvファイルに追加レコード入れて保存
                csv_df = pd.concat([csv_df, df], ignore_index=True)
                csv_df.to_csv(csv_path, index=False, encoding="SHIFT-JIS")

            df = df[['決算期', '発表日', '売上高', '営業益', '経常益', '最終益']]
            df = df.rename(columns={'決算期': 'term', '発表日': 'date', '売上高': 'sales',
                                    '営業益': 'op_income', '経常益': 'ord_income', '最終益': 'net_income'})
        return df, csv_path
    except Exception as e:
        # traceback.print_exc()
        return None, None


def insert_quarterly_df_to_table(code, df, db_file_name, table_name='quarterly_results'):
    """
    3ヵ月決算【実績】情報データフレームをinsert
    """
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        for row in tqdm(df.itertuples(index=False)):
            try:
                # insert quarterly_results
                c.execute(f'INSERT INTO {table_name} (code,term,date,sales,op_income,ord_income,net_income) VALUES(?,?,?,?,?,?,?)',
                          (code, row.term, row.date, row.sales, row.op_income, row.ord_income, row.net_income))
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-dir", "--save_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly',
                    help="quarterly csv file path.")
    ap.add_argument("-b_dir", "--brand_csv_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabuoji3',
                    help="brand code csv file path.")
    ap.add_argument("-u", "--is_update", action='store_const', const=True, default=False,
                    help="table update flag.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    save_dir = args['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    if args['is_update']:
        # 最初に作成したcsvファイル名から銘柄コード取得する
        pbar = tqdm(glob.glob(os.path.join(save_dir, "*.csv")))
        for path in pbar:
            pbar.set_description(path)
            code = os.path.basename(path).split('.')[0]
            # csvに新規レコード追記する
            df, csv_path = download_quarterly_csv(code, csv_path=path)
            insert_quarterly_df_to_table(code, df, db_file_name)
    else:
        # 株価のcsvファイル名から銘柄コード取得する
        pbar = tqdm(glob.glob(os.path.join(args['brand_csv_dir'], '*.csv')))
        for brand_csv in pbar:
            code = str(pathlib.Path(brand_csv).stem)
            pbar.set_description(code)
            df, csv_path = download_quarterly_csv(code, save_dir=save_dir)
            if csv_path is not None:
                insert_quarterly_df_to_table(code, df, db_file_name)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
レーティング情報スクレイピングしてSQLiteに格納
Usage:
    1. （コードからでもできるが）DB Browser for SQLiteで、データベース作成+「raw_ratings」「ratings」テーブル作成
    CREATE TABLE ratings (
        date TEXT, -- 公開日
        code TEXT, -- 銘柄コード
        think_tank TEXT, -- 目標株価を公表した証券会社などの名前
        rating TEXT, -- レーティング
        target REAL, -- 目標株価（調整後）
        PRIMARY KEY(date, code, think_tank)
    );
    CREATE TABLE raw_ratings (
        date TEXT, -- 公開日
       code TEXT, -- 銘柄コード
       think_tank TEXT, -- 目標株価を公表した証券会社などの名前
       rating TEXT, -- レーティング
       target REAL, -- 目標株価（未調整）
       PRIMARY KEY(date, code, think_tank)
    );

    2. 本モジュール実行
    ※変数「db_file_name」は作成したデータベースファイルのパスに変更すること
    $ activate stock
    $ python no_050_rating_data.py
"""
import argparse
import datetime
import os
import re
import sqlite3
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

today = datetime.datetime.today().strftime("%Y-%m-%d")  # '2020-03-07'
str_year = today[:4]
str_month = today[5:7]

# 公開プロキシのファイルあるか
import json
import pathlib
import random
import requests
import warnings
warnings.filterwarnings("ignore")
current_dir = pathlib.Path(__file__).resolve().parent
PROXIES_PATH = os.path.join(current_dir, 'proxies.json')
http_ips, https_ips = [], []
# # 公開プロキシ介すると遅いし失敗することあるのでやめておく
# if os.path.exists(PROXIES_PATH):
#     json_load = json.load(open(PROXIES_PATH, 'r'))
#     http_ips = [s for s in json_load if 'http://' in s if ':3128' in s]
#     https_ips = [s for s in json_load if 'https://' in s if ':3128' in s]


def edit_rating_df(df, yyyy, mm, save_dir):
    """ rating情報データフレームを加工する """
    df.columns = df.loc[0]
    df = df.replace('−', np.nan)
    df = df.dropna(how='any')
    df = df.drop(0).reset_index(drop=True)

    if df.shape[0] > 0:
        df['日付'] = pd.to_datetime(yyyy + '/' + df['日付']).dt.strftime("%Y-%m-%d")

        # この段階でcsvにしておく
        csv_path = os.path.join(save_dir, 'rating_' + yyyy + mm + '.csv')
        df.to_csv(csv_path, index=False, encoding="SHIFT-JIS")

        df = df.rename(columns={'日付': 'date', 'コード': 'code', '市場': 'market', '銘柄名': 'code_name',
                                'シンクタンク': 'think_tank', 'レーティング': 'rating', 'ターゲット': 'target'})

        df['target'] = df['target'].str.replace('円', '')
        df['target'] = df['target'].str.replace('新規', '')
        df['target'] = df['target'].str.replace('継続', '')
        df['target'] = df['target'].str.replace('前後', '')
        df['target'] = df['target'].map(lambda x: re.sub('\.[0-9]+万', '0000', x))  # 小数点切り捨てる
        df['target'] = df['target'].str.replace('万', '0000')
        df['target'] = df['target'].map(lambda x: re.sub("\*$", '', x))

        def _split_concat(df_, s):
            df_not_s = df_[~df_['target'].str.contains(s)]
            if df_not_s.shape[0] < df_.shape[0]:
                df_s = df_[df_['target'].str.contains(s)]
                df_s['target'] = df_s['target'].str.split(s, expand=True)[1]
                return pd.concat([df_s, df_not_s]).sort_index()
            else:
                return df
        df = _split_concat(df, '〜')
        df = _split_concat(df, '→')

        df['target'] = pd.to_numeric(df['target'], errors='coerce')  # target列は数値のみにする
        df = df.dropna(how='any')

    return df.sort_values(by='date', ascending=False)


def download_recent_rating_csv(save_dir):
    """
    トレーダーズ・ウェブ（http://www.traders.co.jp/）から今月のレーティング情報スクレイピングしてinsertする
    ※トレーダーズ・ウェブはスクレイピング許可されてない（はず）。やりすぎると逮捕されかねないので注意
    """
    try:
        url = 'https://www.traders.co.jp/domestic_stocks/domestic_market/attention_rating/attention_rating.asp'

        # Referrer指定して、直前になんのサイトを見ていたかを偽装する。一応、使っているブラウザのUserAgentも明記する。Googleで検索する際に、"my useragent"と入力すると、Googleが今使っているブラウザのUserAgentを教えてくれる
        headers = {"referer": "https://google.com",
                   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}
        if len(http_ips) > 0:
            proxies = {"https": random.choice(http_ips) + "/"}  # プロキシサーバ経由してIPアドレス偽装する
            response = requests.get(url=url, headers=headers, proxies=proxies, verify=False)  # verify=Falseで証明書の警告を無視
        else:
            response = requests.get(url=url, headers=headers)
        html = response.content

        dfs = pd.read_html(html)  # dfs = pd.read_html(url)でもいけるがIP情報など偽装するために requests かます
        for df in dfs:
            if set(df.loc[0].to_list()) == set(['日付', 'コード', '市場', '銘柄名', 'シンクタンク', 'レーティング', 'ターゲット']):
                break
        # df = dfs[40]
        if df.shape[0] > 0:
            df = edit_rating_df(df, str_year, str_month, save_dir)
            if df.shape[0] > 0:
                return df
            else:
                return None
    except Exception as e:
        traceback.print_exc()
        return None


def download_old_rating_csv(yyyy, mm, save_dir):
    """
    トレーダーズ・ウェブ（http://www.traders.co.jp/）から古いレーティング情報スクレイピングしてcsvに保存する
    ※トレーダーズ・ウェブはスクレイピング許可されてない（はず）。やりすぎると逮捕されかねないので注意
    """
    try:  # トレーダーズ・ウェブがエラーになる場合があったのでtryで囲む
        url = f'https://www.traders.co.jp/domestic_stocks/domestic_market/attention_rating/attention_rating_bn.asp?BN={yyyy}{mm}'

        # Referrer指定して、直前になんのサイトを見ていたかを偽装する。一応、使っているブラウザのUserAgentも明記する。Googleで検索する際に、"my useragent"と入力すると、Googleが今使っているブラウザのUserAgentを教えてくれる
        headers = {"referer": "https://google.com",
                   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}
        if len(http_ips) > 0:
            proxies = {"https": random.choice(http_ips) + "/"}  # プロキシサーバ経由してIPアドレス偽装する
            response = requests.get(url=url, headers=headers, proxies=proxies, verify=False)  # verify=Falseで証明書の警告を無視
        else:
            response = requests.get(url=url, headers=headers)
        html = response.content

        dfs = pd.read_html(html)  # dfs = pd.read_html(url)でもいけるがIP情報など偽装するために requests かます
        for df in dfs:
            if set(df.loc[0].to_list()) == set(['日付', 'コード', '市場', '銘柄名', 'シンクタンク', 'レーティング', 'ターゲット']):
                break
        # df = dfs[41]
        if df.shape[0] > 0:
            df = edit_rating_df(df, yyyy, mm, save_dir)
            if df.shape[0] > 0:
                return df
            else:
                return None
    except Exception:
        traceback.print_exc()
        return None


def insert_rating_df_to_table(df, db_file_name, table_names=['raw_ratings', 'ratings']):
    """
    レーティング情報データフレームをinsert
    """
    with sqlite3.connect(db_file_name) as conn:
        c1 = conn.cursor()
        c2 = conn.cursor()
        for row in tqdm(df.itertuples(index=False)):
            try:
                # insert raw_prices
                c1.execute(f'INSERT INTO {table_names[0]} (date,code,think_tank,rating,target) VALUES(?,?,?,?,?)',
                           (row.date, row.code, row.think_tank, row.rating, row.target))
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass
            try:
                # insert prices
                c2.execute(f'INSERT INTO {table_names[1]} (date,code,think_tank,rating,target) VALUES(?,?,?,?,?)',
                           (row.date, row.code, row.think_tank, row.rating, row.target))
            except sqlite3.IntegrityError:  # 重複レコードinsertしたとき
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    ap.add_argument("-dir", "--save_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\traders',
                    help="rating csv file path.")
    ap.add_argument("-is_o", "--is_old", action='store_const', const=True, default=False,
                    help="old data download flag.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    save_dir = args['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    if args['is_old'] == False:
        # 今月のレーティング情報csvファイルスクレイピング
        df = download_recent_rating_csv(save_dir)
        if df is not None:
            insert_rating_df_to_table(df, db_file_name)
    else:
        for yyyy in tqdm(range(2005, int(str_year) + 1)):
            for mm in range(1, 12 + 1):
                yyyy = str(yyyy)
                mm = str(mm).zfill(2)
                # 古いレーティング情報csvファイルスクレイピング
                df = download_old_rating_csv(yyyy, mm, save_dir)
                if df is not None:
                    insert_rating_df_to_table(df, db_file_name)
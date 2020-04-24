#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株式投資メモ<https://kabuoji3.com/>から四本値（日足）と出来高などをスクレイピングしてダウンロード
※全銘柄取ると35時間ぐらい掛かる
※スクレイピングの可/不可はそのサイトのrobots.txtでわかる。
　Yahoo!ファイナンスではスクレイピング禁止 https://support.yahoo-net.jp/PccFinance/s/article/H000011276
　株式投資メモのrobots.txtは下記サイトから確認できる。（robots.txtはサイト直下にある）
　https://kabuoji3.com/robots.txt
　User-Agent: * # 全員にスプレイピングを許可している
　Allow:/       # 「/」にallow（許可）がついています。「/」はルートディレクトリですから、全部スクレイピングして良いということ
　とあるからスクレイピングしてもOK
　http://int-info.com/index.php/scr00/ でもOKと書いているがサーバ負荷かけるからよろしくはない。。。（逮捕される可能性は0ではない）
Usage:
    $ activate stock
    $ python no_020_kabuoji3_csv_download.py  # 全銘柄とる
    $ python no_020_kabuoji3_csv_download.py -o tmp_dir -c 1301 7974  # 指定銘柄だけとる
    $ python no_020_kabuoji3_csv_download.py -o tmp_dir -c 7974 -s_y 2010 -e_y 2020  # 指定銘柄+指定期間だけとる
"""
import argparse
import datetime
import os
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
import util


def download_stock_csv(code_range: list, save_dir: str, start_year: int, end_year: int):
    """
    pandasで株式投資メモの株価ダウンロードして銘柄コードごとにcsv出力
    ※スクレイピングなのでサーバに負荷掛かる。やりすぎないこと
    """
    for code in tqdm(code_range):
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
                pass

        if df_concat is not None:
            df_concat.to_csv(os.path.join(save_dir, str(code) + '.csv'), index=False, encoding="SHIFT-JIS")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default=r'D:\DB_Browser_for_SQLite\csvs\kabuoji3', help="output dir path.")
    ap.add_argument("-s_y", "--start_year", type=int, default=1980, help="search start year.")
    ap.add_argument("-e_y", "--end_year", type=int, default=None, help="search end year.")
    ap.add_argument("-c", "--codes", type=int, nargs='*', default=None, help="brand code list.")
    args = vars(ap.parse_args())

    save_dir = args['output_dir']
    os.makedirs(save_dir, exist_ok=True)

    if args['end_year'] is None:
        today = datetime.datetime.today().strftime("%Y-%m-%d")  # '2020-03-07'
        end_year = today[:4]
    else:
        end_year = args['end_year']

    if args['codes'] is None:
        # codes=list(range(1301, 9998)) は無駄が多いのでDBから銘柄コード全件取る
        brands = np.array(util.check_table('brands'))
        codes = list(brands[:, 0])
    else:
        codes = args['codes']

    download_stock_csv(codes, save_dir, args['start_year'], end_year)
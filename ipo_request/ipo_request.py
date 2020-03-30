#! python3
# -*- coding: utf-8 -*-
"""
IPO銘柄の一覧を取得して申込む
Usage:
    python ipo_list.py
"""

#使用するモジュールのインポート
from modules import kaisya_data as kd
from modules import kaisya_list as kl
from modules import kaisya_csv as kc
from modules import request_ipo as ri

###初期処理###
import os
import sys
import datetime
import pandas as pd

#CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"
password_dir = 'password'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

###メイン処理###
k_codes = input("""★★IPO情報を更新する会社を全て選んでください★★:
            １：楽天証券
            ２：ＳＢＩ証券
            ３：ＳＭＢＣ日興証券
            ４：マネックス証券
            ５：松井証券
            ６：ＧＭＯクリック証券
            ７：野村証券
            ８：ａｕカブコム証券
            ９：大和証券
            １０：東海東京証券
            １１：岡三オンライン証券
            １２：ＤＭＭ．ｃｏｍ証券
            １３：岩井コスモ証券
            １４：ライブスター証券（２銘柄以上の挙動確認）
            １５：ストリーム（ＩＰＯなし）
            １６：みずほ証券
            １７：ＬＩＮＥ証券（ＩＰＯなし）
            １８：ネオモバイル証券
            ９９：ＡＬＬ
番号を入力してください：""").split(",")

#ALL指定の場合は全ての番号をループする（splitでリストになっている）
if k_codes[0] == "99":
    k_codes = list(range(0,19))
    k_codes = [str(k_code) for k_code in k_codes]
else:
    assert set(k_codes) <= set([str(k_code) for k_code in list(range(0,19))]), "\n★★証券会社の番号が間違っています。指定された1-18,99のいずれかを入れて下さい。★★"

# 一応毎回マスタ更新する
print("マスタのファイル更新開始...")
_ = kl.kaisya_list("0",None,output_dir)

#入力した会社情報を順番に取得する
for k_code in k_codes:
    k_data = []
    k_data = kd.kaisya_data(k_code, password_dir)
    print(str(k_data[0][0]) + "のファイル更新開始...")

    #入力した番号の会社のIPOリストを取得
    ipo_list = kl.kaisya_list(k_code,k_data,output_dir)

    #各会社のIPOリストからcsvファイルを作成する
    kc.kaisya_csv(ipo_list,k_data,output_dir)

# ipo申込み
for k_code in k_codes:
    df_regist_ipo = ri.get_request_ipo_info(os.path.join(output_dir, 'master.csv'), os.path.join(output_dir, k_data[0][0]+'.csv'))

    if df_regist_ipo.shape[0] == 0:
        print(f"★★{k_data[0][0]}の購入申込期間中のIPOはありません★★")
        continue

    print(f"\n★★{k_data[0][0]}の購入申込期間中のIPOは以下です★★")
    print(df_regist_ipo)

    ipo_codes = input("\n★★申込むIPOコードをカンマ区切りで入力してください。指定なければすべて申込みます★★:").split(",")
    if ipo_codes == ['']:
        ipo_codes = df_regist_ipo.index.to_list()
        print('すべて申込みます')

    ipo_request = ri.IpoRequest(password_dir)# , CHROMEDRIVER
    for ipo_code in ipo_codes:
        try:
            price = input(f"\n★★{ipo_code}の申込みを行います。価格を指定する場合は入力してください。指定なければストライクプライスで申込みます★★:")
            if price == '':
                print('ストライクプライスで申込みます')
                price = None
            else:
                price = int(price)

            # 各社の申込み処理
            if k_data[0][0] == 'sbi':
                ipo_request.request_sbi_ipo(ipo_code, price)
            else:
                print('申込みメソッドがありません')
                raise Exception

            print(f'\n★★{ipo_code}の申込み成功しました★★')
        except Exception as e:
            print(f'\n★★{ipo_code}の申込み失敗しました。入力したIPOコードか価格が不正か、すでに申込み済みの可能性があります。★★')
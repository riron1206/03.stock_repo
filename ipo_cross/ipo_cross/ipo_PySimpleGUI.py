#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usege:
    $ activate stock
    $ python ipo.py
"""
import argparse
import os
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent  # このファイルのディレクトリの絶対パスを取得
sys.path.append(str(current_dir))
from libs import cross_sbi as cs
from libs import kaisya_csv as kc
from libs import kaisya_data as kd
from libs import kaisya_list as kl
from libs import request_ipo as ri

import PySimpleGUI as sg

# CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, default='output',
                    help="output dir path.")
parser.add_argument("-p", "--password_dir", type=str, default='../password',
                    help="password dir path.")
args = vars(parser.parse_args())
output_dir = args['output_dir']
os.makedirs(output_dir, exist_ok=True)
password_dir = args['password_dir']

# 一応毎回マスタ更新する
#_ = kl.kaisya_list("0", None, output_dir)
sg.popup("マスタのファイル更新しました")


def _request_ipo(k_codes):
    for k_code in k_codes:
        # 入力した会社情報を順番に取得する
        k_data = []
        k_data = kd.kaisya_data(k_code, password_dir)
        print(str(k_data[0][0]) + "のファイル更新開始...")

        try:
            # 入力した番号の会社のIPOリストを取得
            ipo_list = kl.kaisya_list(k_code, k_data, output_dir)

            # 各会社のIPOリストからcsvファイルを作成する
            kc.kaisya_csv(ipo_list, k_data, output_dir)

            # ipo申込むレコード取得
            df_regist_ipo = ri.get_request_ipo_info(os.path.join(output_dir, 'master.csv'), os.path.join(output_dir, k_data[0][0] + '.csv'))
            if df_regist_ipo.shape[0] == 0:
                sg.popup(f"★★{k_data[0][0]}の購入申込期間中のIPOはありません★★")
                continue
            df_regist_ipo = df_regist_ipo.reset_index().set_index('コード', drop=False)

            # GUIのテーブル更新
            window['_Table_'].update(df_regist_ipo.values.tolist())

            ipo_codes = sg.popup_get_text('★★申込むIPOコードをカンマ区切りで入力してください。指定なければすべて申込みます★★', 'Please input ipo code')

            ipo_codes = ipo_codes.split(",")
            if ipo_codes == ['']:
                ipo_codes = df_regist_ipo.index.to_list()
                print('すべて申込みます')
        except Exception:
            sg.popup(f'★★ipoの申込み失敗しました★★')
            continue

        ipo_request = ri.IpoRequest(password_dir)  # , CHROMEDRIVER
        for ipo_code in ipo_codes:
            try:
                price = sg.popup_get_text(f"★★{ipo_code}の申込みを行います。価格を指定する場合は入力してください。指定なければストライクプライスで申込みます★★", 'Please input price')
                if price == '':
                    print('ストライクプライスで申込みます')
                    price = None
                else:
                    price = int(price)

                # 各社の申込み処理
                if k_data[0][0] == 'sbi':
                    ipo_request.request_sbi_ipo(ipo_code, price)
                else:
                    sg.popup('申込みメソッドがありません')
                    raise Exception

                sg.popup(f'★★{ipo_code}の申込み成功しました★★')
            except Exception:
                sg.popup(f'★★{ipo_code}の申込み失敗しました。入力したIPOコードか価格が不正か、すでに申込み済みの可能性があります★★')


#  セクション1 - オプションの設定と標準レイアウト
frame = [[sg.Radio('楽天証券', 1, key='1')],
         [sg.Radio('ＳＢＩ証券', 1, key='2', default=True)],
         [sg.Radio('ＳＭＢＣ日興証券', 1, key='3')],
         [sg.Radio('マネックス証券', 1, key='4')],
         [sg.Radio('松井証券', 1, key='5')],
         [sg.Radio('ＧＭＯクリック証券', 1, key='6')],
         [sg.Radio('野村証券', 1, key='7')],
         [sg.Radio('ａｕカブコム証券', 1, key='8')],
         [sg.Radio('大和証券', 1, key='9')],
         [sg.Radio('東海東京証券', 1, key='10')],
         [sg.Radio('岡三オンライン証券', 1, key='11')],
         [sg.Radio('ＤＭＭ．ｃｏｍ証券', 1, key='12')],
         [sg.Radio('岩井コスモ証券', 1, key='13')],
         [sg.Radio('ライブスター証券', 1, key='14')],
         [sg.Radio('みずほ証券', 1, key='16')],
         [sg.Radio('ネオモバイル証券', 1, key='18')],
         [sg.Radio('ＡＬＬ', 1, key='99')]]

col = [[sg.Button('会社選択', size=(10, 3))]]

layout = [[sg.Frame('★★IPO申込む会社を選んでください★★', frame), sg.Column(col)],
          [sg.Text('★★購入申込期間中のIPO★★', size=(30, 1))],
          [sg.Table(values=[['', '', '', '', '', '']], headings=['コード', '銘柄名', '申込開始', '申込終了', '当選本数', '最大価格'],
                    display_row_numbers=True, auto_size_columns=False, num_rows=15, key='_Table_')]]

# セクション 2 - ウィンドウの生成
window = sg.Window('IPO申込み', layout)

# セクション 3 - イベントループ
while True:
    event, values = window.read()

    if event is None:
        print('exit')
        break

    if event == '会社選択':
        keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16', '18', '99']
        for k_codes in keys:
            if values[k_codes]:
                break

        # ALL指定の場合は全ての番号をループする（splitでリストになっている）
        if k_codes == '99':
            k_codes = [str(k_code) for k_code in list(range(0, 19))]
        else:
            k_codes = [str(k_codes)]

        # 指定した会社のipo申込み
        _request_ipo(k_codes)

# セクション 4 - ウィンドウの破棄と終了
window.close()

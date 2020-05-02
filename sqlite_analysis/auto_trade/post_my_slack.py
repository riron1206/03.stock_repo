#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my Slackに結果メッセージを飛ばす
Usage:
    $ python post_my_slack.py
"""
import os
import requests
import json
import pandas as pd
import yaml

# webhookのエンドポイントのurl
with open(os.path.join('password', 'slack.yml')) as f:
    config = yaml.load(f)
    post_url = config['post_url']


def post_slack(name, text):
    requests.post(
        post_url,
        data=json.dumps(
            {"text": text,
             "username": name,
             "icon_emoji": ":python:"}))


df = pd.read_csv(r'input\auto_order.csv', encoding='shift-jis')

date = df['シグナル日'][0]
codes = df['証券コード'].to_list()
str_codes = map(str, codes)  # 格納される数値を文字列にする
str_codes = ', '.join(str_codes)  # リストを文字列にする
text = f"{str(date)} にシグナルが出た銘柄は {len(codes)} 件です。銘柄コードは出来高が多い順に {str_codes} です。"
# print(text)
post_slack('株価シグナル情報', text)
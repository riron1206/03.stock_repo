#! python3
# -*- coding: utf-8 -*-
"""
価格ドットコムからseleniumでIPO情報を取得してcsvファイルに出力する
Usage:
    $ activate tfgpu20
    $ python get_ipo_csv.py
"""
import os
import datetime
import pathlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from selenium import webdriver

class IpoData():
    def __init__(self, url="https://kakaku.com/stock/ipo/schedule/", chromedriver=None):
        self.url = url # デフォルトは価格ドットコムで固定
        self.chromedriver = chromedriver # seleniumのドライバ

    def get_html_tables(self):
        """
        seleniumで画面アクセスしてtableタグのデータをすべて取得
        """
        if self.chromedriver is not None:
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument('--headless')
            driver = webdriver.Chrome(self.chromedriver, options=options)
        else:
            from selenium.webdriver.firefox.options import Options
            options = Options()
            options.add_argument('--headless')
            driver = webdriver.Firefox(options=options)

        driver.get(self.url)
        dfs = pd.read_html(driver.page_source, header = 0)
        driver.quit()
        return dfs

    def _get_price_max(self, row):
        """ price列の値が大きいほうを返す """
        if row['price_max'] is not None:
            row['price_min'] = row['price_max']
        return row['price_min']

    def get_ipo_csv(self, dfs, output_csv='output/ipo.csv'):
        """
        取得したIPO情報をcsvファイルに出力する
        Args:
            dfs:get_html_tables()の返り値
            output_csv:出力するcsvファイルのパス
        """
        ipo_df = None
        for i,df in enumerate(dfs):
            # マスターとなる価格ドットコムのIPO情報
            if 'kakaku' in self.url:
                if i == 0:
                    df = df.iloc[:, [3,1,4,5,6]]
                    df.columns = ['code', 'name', 'entry', 'tickets', 'price']

                    # 銘柄ID
                    df['code'] = df['code'].str[-4:].astype(int)

                    # 期間
                    df['entry'] = df['entry'].str.replace('まもなく申込', '')
                    df['entry'] = df['entry'].str.replace('申込期間中', '')
                    df['entry'] = df['entry'].str.replace('申込期間終了', '')
                    df['entry'] = df['entry'].str.replace('月', '/')
                    df['entry'] = df['entry'].str.replace('日', '')

                    _df = df['entry'].str.split('～', expand=True)
                    _df = _df.rename(columns={0:'entry_start', 1:'entry_end'})
                    now_year = str(datetime.date.today().year)
                    _df['entry_start'] = pd.to_datetime(now_year+'/'+_df['entry_start'])
                    _df['entry_end'] = pd.to_datetime(now_year+'/'+_df['entry_end'])

                    df = pd.concat([df, _df], axis=1)
                    df = df.drop(['entry'], axis=1)

                    # 当選本数
                    df['tickets'] = df['tickets'].str.replace(',', '')
                    df['tickets'] = df['tickets'].str.replace('本', '')
                    df['tickets'] = df['tickets'].astype(int)

                      # 1株の価格　最大値を採用する
                    df['price'] = df['price'].str.replace(',', '')
                    df['price'] = df['price'].str.replace('円', '')
                    df['price'] = df['price'].str.replace('仮', '')
                    _df = df['price'].str.split('～', expand=True)
                    _df = _df.rename(columns={0:'price_min', 1:'price_max'})
                    price_max = _df.apply(self._get_price_max, axis=1)
                    df['price'] = price_max
                    df = df.replace('-', np.nan)

                    df = df.sort_values(by=['code'], ascending=True)
                    ipo_df = df[['code', 'name', 'entry_start', 'entry_end', 'tickets', 'price']]
                    ipo_df.to_csv(output_csv, encoding="shift_jis", header=None, index=False)
                    print("INFO: save file. [{}] {}".format(output_csv, ipo_df.shape))
                    break

            # 楽天のIPO銘柄ID-銘柄名
            if 'rakuten' in self.url:
                if '抽選日' in df.columns:
                    df = df.iloc[:, [1,3]]
                    df.columns = ['code', 'name']
                    # 銘柄ID
                    df['code'] = df['code'].astype(int)
                    df = df.drop_duplicates()
                    ipo_df = df.sort_values(by=['code'], ascending=True)
                    ipo_df.to_csv(output_csv,encoding="shift_jis", header=None, index=False)
                    break

            # 松井証券のIPO銘柄ID-銘柄名
            if 'matsui' in self.url:
                if '購入申込期間' in df.columns:
                    df = df.iloc[1:, [2,1]]
                    df.columns = ['code', 'name']
                    # 銘柄ID
                    df['code'] = df['code'].str[-4:].astype(int)
                    df = df.drop_duplicates()
                    ipo_df = df.sort_values(by=['code'], ascending=True)
                    ipo_df.to_csv(output_csv,encoding="shift_jis", header=None, index=False)
                    break

            #ａｕカブコム証券のIPO銘柄ID-銘柄名
            if 'kabu.com' in self.url:
                if '期間' in df.columns:
                    df = df.iloc[:, [0,1]]
                    df.columns = ['code', 'name']
                    # 銘柄ID
                    df['code'] = df['code'].astype(int)
                    df = df.drop_duplicates()
                    ipo_df = df.sort_values(by=['code'], ascending=True)
                    ipo_df.to_csv(output_csv,encoding="shift_jis", header=None, index=False)
                    break

        return ipo_df

if __name__ == '__main__':
    #CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"
    os.makedirs('output', exist_ok=True)
    ipodata = IpoData()#chromedriver=CHROMEDRIVER
    dfs = ipodata.get_html_tables()
    df = ipodata.get_ipo_csv(dfs)

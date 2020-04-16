#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
銘柄番号指定してseleniumでipoを申込む
参考:https://papa111.com/2018/11/11/post-277/
Usage:
    call activate stock
    call python regist_ipo.py
    pause
"""
import datetime
import os
from time import sleep

import pandas as pd
import yaml
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import Select


def _compare_date(row, now):
    # str→datetime.datetime
    start = datetime.datetime.strptime(row[2], '%Y-%m-%d')
    end = datetime.datetime.strptime(row[3], '%Y-%m-%d')

    # datetime.datetime→datetime.date
    start = datetime.date(start.year, start.month, start.day)
    end = datetime.date(end.year, end.month, end.day)

    if start <= now <= end:
        # print(start, now, end)
        return end
    else:
        return None


def get_request_ipo_info(master_csv: str, kaisya_csv: str):
    """
    ブックビル申込中のIPOの銘柄コードの情報取得
    Args:
        master_csv:ipoのマスター情報csv path
        kaisya_csv:各会社のipo情報csv path
    """
    # ipoのマスター情報csv
    df_master = pd.read_csv(master_csv, encoding="shift_jis", header=None)

    # 各会社のipo情報csv
    df_kaisya = pd.read_csv(kaisya_csv, encoding="shift_jis", header=None)
    # 1列目のipo codeだけにする
    df_kaisya = df_kaisya[[0]]

    # ipoのマスターと各会社のipo情報を内部結合
    df_merge = pd.merge(df_master, df_kaisya, on=0)

    # ブックビル申込中（現在時刻の間）のレコードのみにする
    now = datetime.date.today()
    # now = datetime.date(2020, 3, 15)# テスト用
    df_merge['within_date'] = df_merge.apply(_compare_date, now=now, axis='columns')
    df_merge = df_merge.dropna(subset=['within_date'])
    df_merge = df_merge.drop(['within_date'], axis=1)
    # print(df_merge)
    df_merge.columns = ['コード', '銘柄名', '申込開始日', '申込終了日', '当選本数', '最大価格']
    df_merge = df_merge.set_index('コード')
    return df_merge


class IpoRequest():
    def __init__(self, password_dir, chromedriver=None):
        # パスワードディレクトリ
        self.password_dir = password_dir

        # seleniumのドライバ
        if chromedriver is not None:
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument('--headless')
            self.driver = webdriver.Chrome(chromedriver, options=options)
        else:
            from selenium.webdriver.firefox.options import Options
            options = Options()
            options.add_argument('--headless')
            self.driver = webdriver.Firefox(options=options)

    def request_sbi_ipo(self, code, price=None):
        """
        銘柄番号指定してseleniumでSBIのipoを申込む
        Args:
            code:ipoの銘柄番号
            price:入札する価格。Noneならストライクプライスで入れる
        """
        ######## sbiログイン情報 ########
        with open(os.path.join(self.password_dir, 'sbi.yml')) as f:
            config = yaml.load(f)
            sbi_user_name = config['sbi_USER_ID']
            sbi_login_pass = config['sbi_PASSWORD']
            sbi_torihiki_pass = config['sbi_torihiki_pass']
        #################################
        # sbiのサイト
        self.driver.get(r"https://www.sbisec.co.jp/ETGate")
        # login
        self.driver.find_element_by_name("user_id").send_keys(sbi_user_name)
        self.driver.find_element_by_name("user_password").send_keys(sbi_login_pass)
        self.driver.find_element_by_name("ACT_login").click()
        # 3秒待つ
        sleep(3)
        # トップページのIPO・POのリンクからIPO登録画面に移動
        self.driver.find_element_by_link_text('IPO・PO').click()
        sleep(1)
        # 新規上場株式ブックビルディング/購入意志表示ボタンクリック
        self.driver.find_element_by_xpath('/html/body/div[4]/div/table/tbody/tr/td[1]/div/div[10]/div/div/a/img').click()
        sleep(1)
        try:
            # 指定codeの申込ボタンクリック
            self.driver.find_element_by_xpath(f"//a[contains(@href,'{code}')]/img").click()
            sleep(1)
            self.driver.find_element_by_name("suryo").send_keys("100") # 100株しか申し込めない

            if price is None:
                # ストライクプライス
                self.driver.find_element_by_id("strPriceRadio").click()
            else:
                # 値段指定あれば価格のプルダウン使う
                self.driver.find_element_by_id("kakakuRadio").click()
                kakaku_element = self.driver.find_element_by_name('kakaku')
                kakaku_select_element = Select(kakaku_element)
                kakaku_select_element.select_by_value(f"{price}")

            self.driver.find_element_by_name("tr_pass").send_keys(sbi_torihiki_pass)
            self.driver.find_element_by_name("order_kakunin").click() # 購入確認
            self.driver.find_element_by_name("order_btn").click() # 購入実行
        except NoSuchElementException:
            # 申込ボタンがない場合
            print('すでに申込済みです')
            raise # 呼び出し元に送信
        finally:
            self.driver.quit() # webページを閉じる

if __name__ == '__main__':
    # IPO_register
    code = 4496
    price = 1350.0
    password_dir = 'password'
    #CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"
    ipo_request = IpoRequest(password_dir)#CHROMEDRIVER
    ipo_request.request_sbi_ipo(code, price)

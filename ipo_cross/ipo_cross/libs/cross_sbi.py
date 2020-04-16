#! python3
# -*- coding: utf-8 -*-
"""
cross_sbi.py - ＳＢＩ証券でクロス取引を行う
引数は会社コードと会社データとする
"""
import os
import re
import sys
# import traceback
from time import sleep

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs4
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from tqdm import tqdm


class CrossSbi():
    """
    sbiでクロス取引
    ※クロス取引:
        現物買いと信用売り※（空売り）を、同じ株数・同じ値段で同時におこなうことを指します。
        同じ金額で買いと売りをおこなうので、損益は0円
    """
    def __init__(self, k_data, chromedriver=None):
        self.k_data = k_data

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

        # SBIにログインする
        self.driver.get(self.k_data[1][0])
        self.driver.find_element_by_name("user_id").send_keys(self.k_data[0][2])
        self.driver.find_element_by_name("user_password").send_keys(self.k_data[0][3])
        self.driver.find_element_by_name("ACT_login").click()
        sleep(2)

    def cancel_cross(self):
        """ sbiで注文したクロス取引を取り消す """
        self.driver.find_element_by_xpath("//div[@id='link02']//img[@title='取引']").click()
        sleep(0.5)
        self.driver.find_element_by_xpath("/html/body/div[1]/table/tbody/tr/td[1]/div[1]/table/tbody/tr/td[5]/a").click()

        for i in tqdm(self.driver.find_elements_by_link_text("取消")):
            self.driver.find_element_by_link_text("取消").click()
            self.driver.find_element_by_name("trade_pwd").send_keys(self.k_data[0][4])
            self.driver.find_element_by_name("ACT_place").click()
            self.driver.find_element_by_link_text("取消・訂正").click()
        self.driver.quit()

    def order_cross(self, order_count, order_lists, order_type, output_dir, buy_type):
        """ sbiでクロス取引を注文する """
        for order_list in tqdm(order_lists):
            # 注文画面に移動し注文する
            sleep(3)
            self.driver.find_element_by_xpath("//div[@id='link02']//img[@title='取引']").click()
            sleep(3)
            self.driver.find_element_by_name("stock_sec_code").clear()
            self.driver.find_element_by_name("stock_sec_code").send_keys(order_list[0])

            # 売買条件を設定して信用売り/信用買い
            self.driver.find_element_by_id(buy_type).click()

            # 注文株数を入力する
            self.driver.find_element_by_name("input_quantity").clear()
            self.driver.find_element_by_name("input_quantity").send_keys(order_list[1])

            # 値段指定あれば価格のプルダウン使う
            order = self.driver.find_element_by_name("input_market")
            try:
                Select(order).select_by_value("TKY")  # 東証
            except Exception:
                try:
                    Select(order).select_by_value("NGY")  # 名証
                except Exception:
                    try:
                        Select(order).select_by_value("FKO")  # 福証
                    except Exception:
                        try:
                            Select(order).select_by_value("SPR")  # 札証
                        except Exception:
                            pass

            # 注文条件に寄成を選択する
            order = self.driver.find_element_by_name("nariyuki_condition")
            Select(order).select_by_value("Y")  # 寄成

            # 期間に当日中を選択（デフォルトで設定されている）
            try:
                self.driver.find_elements_by_name("payment_limit")[0].click()  # 当日中
                self.driver.find_elements_by_name("hitokutei_trade_kbn")[1].click()  # 特定預り
            except Exception:
                pass

            # 取引種別を設定する（制度 or 一般）
            self.driver.find_elements_by_name("payment_limit")[order_type].click()

            # パスワード入力、注文確認画面を省略、注文
            self.driver.find_element_by_name("trade_pwd").send_keys(self.k_data[0][4])
            self.driver.find_element_by_name("skip_estimate").click()
            sleep(0.5)
            self.driver.find_element_by_css_selector("[alt='注文発注']").click()
            sleep(1)
            # 注文後、エビデンス取得のためスクレイピングを行う
            res = self.driver.page_source
            soup = bs4(res, "lxml")
            # 「注文受付」の表示で注文結果を確認する（注文受付から始まるh2タグを取得できれば注文成功とする）
            if soup.find("h2", text=re.compile("^注文受付")) is not None:
                try:
                    table = soup.find(class_="md-l-table-01")
                    result_list = []
                    abc = table.find_all("td")
                    code = abc[4].text.replace("\n", "").strip().replace("\xa0", "")
                    meigara = abc[5].text.replace("\n", "").strip().replace("\xa0", "")
                    syubetsu = abc[3].text.replace("\n", "").strip().replace("\xa0", "")
                    volume = abc[7].text.replace("\n", "").strip().replace("\xa0", "")
                    volume = re.sub('[\D]', '', volume)
                    joken = abc[8].text.replace("\n", "").strip().replace("\xa0", "")

                    result_list.append([code, meigara, syubetsu, volume, joken])

                    with open(os.path.join(output_dir, "cross_result.csv"), "a") as f:
                        for d in result_list:
                            f.write("{},{},{},{},{},{},{}\n".format(order_count, d[0], d[1], d[2], d[3], d[4], self.k_data[0][0]))
                except Exception:
                    print("{}の結果取得に失敗しました".format(order_list[0]))
            else:
                print("{}の注文に失敗している可能性があります。確認してください".format(order_list[0]))
                # traceback.print_exc()
            sleep(2)
        self.driver.quit()
        return


def order_main(k_data, input_dir, output_dir):
    # ###################################ここから各社共通################################### #
    # 出力ファイルを読込みヘッダがなければヘッダを作成
    # ヘッダがある場合は処理継続
    try:
        df_result = pd.read_csv(os.path.join(output_dir, "cross_result.csv"), encoding="shift_jis")
        # 過去の注文回数を確認
        order_count = np.max(df_result["注文回数"].astype(int))
        # 初注文の場合は今回注文回数に1をセット
        if order_count == "nan":
            order_count = 1
        else:
            order_count += 1
    except Exception:
        with open(os.path.join(output_dir, "cross_result.csv"), "a") as f:
            f.write("注文回数,銘柄コード,銘柄名,取引内容,株数,執行条件,注文会社\n")
            # 初注文の場合は今回注文回数に1をセット
            order_count = 1

    # 注文票（制度、一般）のcsvファイルを、リストデータとして読込む（注文がない場合はスキップ）
    sedo_skip = "0"  # 初期化
    ippan_skip = "0"  # 初期化
    try:
        order_sedo = pd.read_csv(os.path.join(input_dir, "order_sedo.csv"), encoding="shift_jis", header=None).values.tolist()
    except Exception:
        print("制度信用クロスは注文がないのでスキップします")
        sedo_skip = "1"

    try:
        order_ippan = pd.read_csv(os.path.join(input_dir, "order_ippan.csv"), encoding="shift_jis", header=None).values.tolist()
    except Exception:
        print("一般信用クロスは注文がないのでスキップします")
        ippan_skip = "1"

    if sedo_skip == "1" and ippan_skip == "1":
        print("注文がないので処理を終了します")
        sys.exit()
    else:
        pass
    # ###################################ここまで各社共通################################### #
    # 制度信用取引の注文を行う
    if sedo_skip == "1":
        pass
    else:
        print("{}で制度信用クロスを行います".format(k_data[0][0]))
        # order_typeとbuy_typeは会社によって変えてください
        order_type = 0
        CrossSbi(k_data).order_cross(order_count, order_sedo, order_type, output_dir, r"shinU")

    # 一般信用取引の注文を行う
    if ippan_skip == "1":
        pass
    else:
        print("{}で一般信用クロスを行います".format(k_data[0][0]))
        # order_typeとbuy_typeは会社によって変えてください
        order_type = 1
        CrossSbi(k_data).order_cross(order_count, order_ippan, order_type, output_dir, r"shinU")

    # ##################################ここから各社共通################################### #
    # 制度信用取引の注文を行う
    # 信用売りの結果リストを読込む
    # オーダーとリザルトでリスト構成が異なるため臨時リストを作成し合わせる
    rinji_list = []
    order_type = 0
    order_result = pd.read_csv(os.path.join(output_dir, "cross_result.csv"), encoding="shift_jis", header=None).values.tolist()

    # 過去の注文は無視する
    for result in order_result:
        if result[0] == str(order_count):
            if re.match("\d{4}", result[1]):
                rinji_list.append([result[1], result[4]])
            else:
                pass
        else:
            pass

    # order_typeとbuy_typeは会社によって変えてください
    CrossSbi(k_data).order_cross(order_count, rinji_list, order_type, output_dir, r"shinK")
    # ###################################ここまで各社共通################################### #

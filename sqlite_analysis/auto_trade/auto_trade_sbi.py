#! python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs4
import datetime
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
from time import sleep
import re
import os
from selenium.webdriver.firefox.options import Options
import traceback
from tqdm import tqdm
import yaml

today = datetime.datetime.today()

password_dir = 'password'
input_dir = 'input'

csv = os.path.join(input_dir, "order.csv")  # 本番用
# csv = r'input\test_order.csv'  # テスト用

df = pd.read_csv(csv, encoding="shift_jis")
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
# print(df.columns[0])

# run_day = today - datetime.timedelta(days=7)  # 何日前から実行する場合
run_day = df[df.columns[0]].iloc[-1]  # 最後の日だけ実行する
df = df[df[df.columns[0]] >= run_day]

print('run signal day:', run_day)
print(df)
order_lists = df.fillna(0).values.tolist()

######## sbiログイン情報 ########
with open(os.path.join(password_dir, 'sbi.yml')) as f:
    config = yaml.load(f)
    sbi_USER_ID = config['sbi_USER_ID']
    sbi_PASSWORD = config['sbi_PASSWORD']
    sbi_torihiki_pass = config['sbi_torihiki_pass']
#################################

# SBI証券にログイン
options = Options()
options.headless = True
driver = webdriver.Firefox(options=options)
driver.get("https://www.sbisec.co.jp/ETGate")
driver.find_element_by_name("user_id").send_keys(sbi_USER_ID)
driver.find_element_by_name("user_password").send_keys(sbi_PASSWORD)
driver.find_element_by_name("ACT_login").click()
driver.implicitly_wait(4.5)

for order_list in tqdm(order_lists):
    # 注文日が入っている注文済みの銘柄スキップ
    if order_list[12] == 0:

        # 注文前画面に移動
        driver.find_element_by_xpath("//div[@id='link02']//img[@title='取引']").click()
        # 新規と手仕舞いで処理変更
        if order_list[3] == "新規":  # 新規注文（現物買い・現物売り・信用買い・信用売り）
            # 銘柄コード
            driver.find_element_by_name("stock_sec_code").clear()
            # driver.find_element_by_name("stock_sec_code").send_keys(str(order_list[2]))
            driver.find_element_by_name("stock_sec_code").send_keys(int(order_list[2]))

            # 取引売買条件を設定
            if order_list[4] == "現物買い":
                driver.find_element_by_id("genK").click()

            elif order_list[4] == "現物売り":
                driver.find_element_by_id("genU").click()

            elif order_list[4] == "信用買い":
                driver.find_element_by_id("shinK").click()

            elif order_list[4] == "信用売り":
                driver.find_element_by_id("shinU").click()

            else:
                print("{}の取引売買条件が不正です".format(order_list[2]))

            # 銘柄検索
            driver.find_element_by_name("ACT_order").click()

            # 銘柄検索後、スクレイピング
            res = driver.page_source
            soup = bs4(res, "lxml")
            # 「注文受付」の表示で注文結果を確認する（「注文入力」なら失敗とする）
            if soup.find("h2", text=("注文入力")):
                print("{}の検索に失敗している可能性があります".format(order_list[2]))
                print(soup.find("p", class_="bold").text)
            else:
                ################################ここから注文画面################################
                # 注文条件１（OCO、IFDは省略）
                if order_list[7] == "通常":  # 通常／逆指値
                    driver.find_element_by_id("stocktype_normal").click()
                    # 注文数
                    driver.find_element_by_id("input_quantity").clear()
                    driver.find_element_by_id("input_quantity").send_keys(str(order_list[6]))
                    # 注文関数呼出し

                    if order_list[8] == "指値":  # 指値
                        driver.find_element_by_id("sashine").click()
                        # 注文条件３（条件なし・寄指・引成・不成・IOC成）
                        #order = driver.find_element_by_name("sasine_condition")
                        #Select(order).select_by_value(" ") #条件なし
                        # 指値金額指定
                        driver.find_element_by_id("input_price").clear()
                        driver.find_element_by_id("input_price").send_keys(str(order_list[9]))

                    elif order_list[8] == "成行":  # 成行
                        driver.find_element_by_id("nariyuki").click()
                        # 注文条件３（条件なし・寄成・引成・IOC成）
                        #order = driver.find_element_by_name("nariyuki_condition")
                        #Select(order).select_by_value("Y") #寄成

                    elif order_list[8] == "逆指値":  # 逆指値
                        driver.find_element_by_id("gyakusashine_gsn2").click()
                        driver.find_element_by_id("input_trigger_price").clear()
                        driver.find_element_by_id("input_trigger_price").send_keys(str(order_list[9]))

                        driver.find_element_by_id("gyakusashine_nariyuki").click()

                        # 注文条件４（条件なし・引成）
                        #order = driver.find_element_by_id("gyakusashine_nariyuki")
                        #Select(order).select_by_value("N") #条件なし

                    else: #上記以外の場合
                        print("注文条件が不正です（指値・成行・逆指値以外）")

                    # 信用取引区分（制度、一般、日計）
                    # 取引売買条件を設定
                    #if order_list[4] == "信用買い" \
                    #or order_list[4] == "信用売り":
                    #    driver.find_elements_by_name("ifdoco_payment_limit")[0].click() # 制度
                    #else:
                    #    pass

                elif order_list[7] == "IFDOCO":  # IFDOCO
                    driver.find_element_by_id("stocktype_ifdoco").click()
                    # 注文数
                    driver.find_element_by_id("ifdoco_input_quantity").clear()
                    driver.find_element_by_id("ifdoco_input_quantity").send_keys(str(order_list[6]))
                    # 注文関数呼出し

                    if order_list[8] == "指値":  # 指値
                        driver.find_element_by_id("sashine_ifdoco_u").click()
                        # 注文条件３（条件なし・寄指・引成・不成・IOC成）
                        #order = driver.find_element_by_name("sasine_condition")
                        #Select(order).select_by_value(" ") #条件なし
                        # 指値金額指定
                        driver.find_element_by_id("ifoco_input_price").clear()
                        driver.find_element_by_id("ifoco_input_price").send_keys(str(order_list[9]))

                    elif order_list[8] == "成行":  # 成行
                        driver.find_element_by_id("nariyuki_ifdoco_u").click()
                        # 注文条件３（条件なし・寄成・引成・IOC成）
                        order = driver.find_element_by_name("ifoco_nariyuki_condition")
                        Select(order).select_by_value("Y")  # 寄成

                    elif order_list[8] == "逆指値":  # 逆指値
                        driver.find_element_by_id("gyakusashine_ifdoco").click()
                        driver.find_element_by_id("ifoco_input_trigger_price").clear()
                        driver.find_element_by_id("ifoco_input_trigger_price").send_keys(str(order_list[9]))

                        driver.find_element_by_id("gyakusashine_nariyuki_ifdoco_u").click()

                        # 注文条件４（条件なし・引成）
                        #order = driver.find_element_by_id("gyakusashine_nariyuki")
                        #Select(order).select_by_value("N") #条件なし

                        #期間指定(最長の2週間後)
                        order = driver.find_element_by_name("doneoco_limit_in")
                        Select(order).select_by_index("13")

                    else:  # 上記以外の場合
                        print("注文条件が不正です（指値・成行・逆指値以外）")

                    # OCO注文
                    driver.find_element_by_id("doneoco1_input_price").send_keys(str(order_list[10]))
                    driver.find_element_by_id("doneoco2_input_trigger_price").send_keys(str(order_list[11]))
                    driver.find_element_by_id("nariyuki_ifdoco").click()

                else:
                    print("{}の注文条件が不正です（通常／逆指値、IFDOCO以外）".format(order_list[2]))

                # 当日中、特定預かり（デフォルト）
                # 信用は特定預かり指定不可のためtry文
                try:
                    driver.find_elements_by_name("payment_limit")[0].click()  # 当日中
                    driver.find_elements_by_name("hitokutei_trade_kbn")[1].click()  # 特定預り
                except:
                    pass

                # パスワード入力、注文
                driver.find_element_by_name("trade_pwd").send_keys(sbi_torihiki_pass)
                driver.find_element_by_name("skip_estimate").click()
                driver.find_element_by_xpath("//img[@alt='注文発注']").click()

                # 注文後、スクレイピング
                res = driver.page_source
                soup = bs4(res, "lxml")
                # 「注文受付」の表示で注文結果を確認する（注文受付から始まるh2タグを取得できれば注文成功とする）
                if soup.find("h2", text=re.compile("^注文受付")) is not None:
                    df = pd.read_csv(os.path.join(csv), encoding="shift_jis")
                    _df = df.astype("str")
                    for index, row in _df.iterrows():
                        if (row.loc["注文日"] == "nan") and (row.loc["証券コード"] == str(order_list[2])):
                            df.loc[index, ["注文日"]] = today.date()
                            df.to_csv(os.path.join(csv), encoding="SHIFT-JIS", index=False)  # 出力ファイル更新
                        else:
                            pass

                    else:
                        pass

                else:
                    print("{}の注文に失敗している可能性があります".format(order_list[2]))
                    print(soup.find("p", class_="bold").text)
                    ANS = "1"
                    traceback.print_exc()

        elif order_list[3] == "手仕舞い":  # 手仕舞い注文（信用返済（買い・売り））
            pass

        else:
            print("{}の注文内容が不正です（新規・手仕舞い以外）処理を終了します".format(order_list[2]))

    # 注文済みの銘柄スキップ
    else:
        pass

driver.quit()  # webページを閉じる

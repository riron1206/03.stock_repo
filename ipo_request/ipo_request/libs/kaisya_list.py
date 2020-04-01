#! python3
# -*- coding: utf-8 -*-
#kaisya_list.py - 各会社のIPOリストを取得する
#引数は会社コードと会社データとする

#使用するモジュールのインポート
from bs4 import BeautifulSoup as bs4
from selenium import webdriver
from openpyxl import load_workbook
from selenium.webdriver.firefox.options import Options
#from selenium.webdriver.chrome.options import Options
#CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"

import requests
import lxml
import re
import xlsxwriter
import pandas as pd
import openpyxl as px
import time

import os, sys, pathlib
current_dir = pathlib.Path(__file__).resolve().parent # このファイルのディレクトリの絶対パスを取得
sys.path.append( str(current_dir) )
import get_ipo_csv

#指定された会社のIPOリストを取得する
def kaisya_list(k_code,k_data,output_dir):
    ipo_list = []
    #ヘッドレスモードのオプションを設定する
    options = Options()
    #options.headless = True
    options.add_argument('--headless')
    if k_code == "0": #マスタ
        ipodata = get_ipo_csv.IpoData()#chromedriver=CHROMEDRIVER
        dfs = ipodata.get_html_tables()
        ipodata.get_ipo_csv(dfs, output_csv=os.path.join(output_dir, 'master.csv'))

    elif k_code == "1": #楽天証券
        #seleniumで情報を取得し、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])
        url = driver.page_source
        soup = bs4(url,"lxml")

        #IPO情報を取得し、リストを作成する
        table = soup.find_all("table",class_="s1-tbl-data01 s1-tbl--width-full fs-xs")
        for tr_loops in table[0].find_all("tr"):
            td_loop = tr_loops.find_all("td")
            try:
                meigara = td_loop[3].find("a")
                code = td_loop[1]
                meigara = meigara.string.strip()
                code = code.string
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "2": #ＳＢＩ証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        tables = soup.find_all("table")[1:-2]
        for table in tables:
            meigara_code = table.find("p",class_="fl01")
            try:
                if meigara_code is None:
                    pass
                else:
                    meigara_code = meigara_code.string
                    meigara_code = meigara_code.replace("\xa0","")
                    meigara_code = meigara_code.replace("（株）","")
                    meigara = meigara_code
                    code = meigara_code
                    meigara = re.sub("（\d.*","",meigara)
                    code = meigara_code[:-4]
                    code = re.sub('\D', '',code)
                    ipo_list.append([code,meigara])
            except:
                pass

    elif k_code == "3": #ＳＭＢＣ日興証券
        #会社ページ情報を取得し、ログイン、ページ移動後、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])

        #ログイン
        driver.find_element_by_id("padInput0").send_keys(k_data[0][1])
        driver.find_element_by_id("padInput1").send_keys(k_data[0][2])
        driver.find_element_by_id("padInput2").send_keys(k_data[0][3])
        driver.find_element_by_name("logIn").click()
        time.sleep(10)

        #IPOの一覧ページに移動する
        driver.find_element_by_xpath("/html/body/div[3]/div[2]/table/tbody/tr/td[3]/div[3]/div[2]/div/table/tbody/tr/td/div[1]/table/tbody/tr/td[2]/span/a").click()
        time.sleep(10)
        url = driver.page_source
        soup = bs4(url,"lxml")

        #会社ページからIPO情報を取得し、リストを作成する
        td = soup.find_all("td",class_="con_15_hd_bg")
        for td_loops in td:
            try:
                meigara = td_loops.find("span",class_="txt_b03")
                code = td_loops.find("span",class_="txt_01_1")
                meigara = meigara.string.replace('　', ' ')
                code = code.string[-4:]
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "4": #マネックス証券
        #会社ページ情報を取得し、ログイン、ページ移動後、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])

        #ログイン
#        driver.find_element_by_name("loginid").send_keys(k_data[0][2])
#        driver.find_element_by_name("passwd").send_keys(k_data[0][3])
#        driver.find_element_by_class_name("text-button.ml-100").click()
#        time.sleep(10)

        #IPOの一覧ページに移動する
        driver.find_element_by_link_text("IPO 取扱銘柄").click()
        time.sleep(3)
#        driver.find_element_by_css_selector("div.row:nth-child(14) > div:nth-child(1) > h3:nth-child(2) > a:nth-child(1)").click()
#        time.sleep(10)
        url = driver.page_source
        soup = bs4(url,"lxml")

        #会社ページからIPO情報を取得し、リストを作成する
        #銘柄情報のあるテーブルを取得後、その中のtdタグを全て取得する
        table = soup.find_all("table",class_="table-block type-bg-02 type-border-01")
        td_loops = table[0].find_all("td",rowspan="2")
        for td_loop in td_loops:
            try:
                meigara = td_loop.find("a").string.replace("　"," ")
                code = td_loop.text.strip()[-4::]
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "5": #松井証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        table = soup.find_all("table",class_="m-tbl")
        tr_loops = table[0].find_all("tr")
        for tr_loop in tr_loops:
            try:
                td_loops = tr_loop.find_all("td")
                meigara = td_loops[1].find_all("a")[1]
                code = td_loops[2]
                meigara = meigara.text
                code = code.text.strip()[-4::]
                ipo_list.append([code,meigara])
            except:
                pass
    elif k_code == "6": #ＧＭＯクリック証券　未実装
        pass
    elif k_code == "7": #野村証券
        #会社ページ情報を取得し、ログイン、ページ移動後、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])

        #ログイン
        driver.find_element_by_name("btnCd").send_keys(k_data[0][1])
        driver.find_element_by_name("kuzNo").send_keys(k_data[0][2])
        driver.find_element_by_name("gnziLoginPswd").send_keys(k_data[0][3])
        driver.find_element_by_name("buttonLogin").click()
        time.sleep(10)

        #IPOの一覧ページに移動する
        driver.find_element_by_link_text('取引').click()
        time.sleep(10)
        driver.find_element_by_link_text('IPO/PO').click()
        time.sleep(10)
        url = driver.page_source
        soup = bs4(url,"lxml")

        #会社ページからIPO情報を取得し、リストを作成する
        #銘柄情報のあるspanタグを全て取得する
        spans = soup.find_all("span",class_="horizontal-rhythm")
        for span in spans:
            try:
                meigara = span
                code = span.find(class_="txt-code")
                meigara = meigara.text[5::].replace("　"," ")
                code = code.text
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "8": #ａｕカブコム証券　未実装
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])
        url = driver.page_source
        soup = bs4(url,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        tables = soup.find_all("table",class_="tbl01")
        for table in tables:
            try:
                code = table.find_all("td")[0]
                meigara = table.find_all("td")[1]
                meigara = meigara.string
                code = code.string
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "9": #大和証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])
        driver.find_element_by_link_text('現在のIPO（新規公開株式）取扱銘柄一覧').click()
        url = driver.page_source
        soup = bs4(url,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        h2s = soup.find_all("h2",class_="hdg-l2")
        for h2 in h2s:
            try:
                meigara = h2.find("a")
                code = h2
                meigara = meigara.string
                code = h2.text[-5:-1]
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "10": #東海東京証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])
        time.sleep(10)
        driver.find_element_by_xpath('/html/body/div[7]/div[1]/div[4]/div[1]/p/a/img').click()
        url = driver.page_source
        soup = bs4(url,"lxml")

        #会社ページからIPO情報を取得し、リストを作成する
        sections = soup.find_all("div",class_="section")
        for section in sections:
            try:
                meigara = section.find("h3",class_="heading")
                code = section.find("tr",class_="odd").find("td")
                meigara = meigara.text.strip().replace("　"," ")[:-3]
                code = code.text
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "11": #岡三オンライン証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        section = soup.find("div",class_="section")
        for th_loop in section.find_all("th",class_="ipoTitle"):
            try:
                meigara = th_loop
                code = th_loop.find("span")
                meigara = meigara.text.replace("　","")
                code = code.text[:4]
                ipo_list.append([code,meigara])
            except:
                pass

    elif k_code == "12": #ＤＭＭ．ｃｏｍ証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        loops = soup.find_all("div",class_="p-ipoTableList")
        for loop in loops:
            try:
                meigara = loop.find("h3",class_="c-subHeading")
                code = loop.find("h3",class_="c-subHeading")
                meigara = meigara.string[:-6]
                code = code.string[-5:-1]
                ipo_list.append([code,meigara])
            except:
                pass

    elif k_code == "13": #岩井コスモ証券
        #会社ページ情報を取得し、ログイン、ページ移動後、bs4オブジェクトに変換する
        driver = webdriver.Firefox(options=options)
        driver.get(k_data[1][1])
        time.sleep(5)

        #ログイン
        driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div/div[2]/div[1]/div[2]/input').send_keys(k_data[0][2])
        driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div/div[2]/div[2]/div[2]/input').send_keys(k_data[0][3])
        driver.find_element_by_class_name("btn.btn-exe").click()
        time.sleep(10)

        #IPOの一覧ページに移動する
        driver.find_element_by_link_text('現物・IPO').click()
        time.sleep(10)
        driver.find_element_by_link_text('需要申告参加').click()
        time.sleep(10)
        url = driver.page_source
        soup = bs4(url,"lxml")

        #会社ページからIPO情報を取得し、リストを作成する
        tables = soup.find_all("table",class_="table table-order")
        tr_loops = tables[1].find_all("tr")
        for tr_loop in tr_loops:
            td_loop = tr_loop.find_all("td")
            try:
                meigara = td_loop[1]
                code = td_loop[0]
                meigara = meigara.string.replace("　"," ")
                code = code.string
                ipo_list.append([code,meigara])
            except:
                pass
        driver.quit()

    elif k_code == "14": #ライブスター証券　未実装
        #会社ページにアクセスし、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        table = soup.find("table",class_="table-block")
        meigara = table.find("td")
        meigara = meigara.text
        meigara = re.sub("（\d{4}.*","",meigara)
        code = table.find("td")
        code = code.text.replace("　","")
        code = re.sub("\D","",code)
        code = code[:4]
        ipo_list.append([code,meigara])

    elif k_code == "15": #ストリーム　未実装
        pass
    elif k_code == "16": #みずほ証券
        #会社ページにアクセスし、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        table = soup.find("table",class_="type1 table_m")
        tr_loops = table.find_all("tr")
        for tr_loop in tr_loops:
            td_loop = tr_loop.find_all("td",class_="center middle")
            try:
                meigara = td_loop[1]
                code = td_loop[1]
                meigara = meigara.text.replace("　"," ")[:-6]
                code = code.text.replace("　"," ")[-5:-1]
                ipo_list.append([code,meigara])
            except:
                pass
    elif k_code == "17": #ＬＩＮＥ証券　未実装
        pass
    elif k_code == "18": #ネオモバイル証券
        #会社ページ情報を取得し、bs4オブジェクトに変換する
        res = requests.get(k_data[1][1])
        soup = bs4(res.content,"lxml")
        #会社ページからIPO情報を取得し、リストを作成する
        loops = soup.find_all("div",class_="tableArea01")
        for loop in loops:
            try:
                tr_loops = loop.find_all("tr")
                meigara = tr_loops[0].find("td")
                code = tr_loops[1].find("td")
                meigara = meigara.text.replace("　"," ")
                code = code.text
                ipo_list.append([code,meigara])
            except:
                pass

    else:
        pass

    return ipo_list


###立会外分売で使用する###
def bunbai_list(bun_data):
    bun_list = []
    #seleniumで情報を取得し、bs4オブジェクトに変換する
    res = requests.get(bun_data[1][1])
    soup = bs4(res.content,"lxml")
    #ネットから立会外分売情報をを取得し、リストを作成する
    list = soup.find("div",class_="mainlist")
    tr_loops = list.find_all("tr")
    for tr_loop in tr_loops:
        try:
            td_loops = tr_loop.find_all("td")
            meigara = tr_loop.find_all("a")[1]
            code = tr_loop.find_all("a")[0]
            day = tr_loop.find("td",class_=("tcenter"))
            hyoka = td_loops[10]
            meigara = meigara.text
            code = code.text
            day = day.text
            hyoka = hyoka.text
            bun_list.append([code,meigara,day,hyoka])

        except:
            pass

    return bun_list
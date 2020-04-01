#! python3
# -*- coding: utf-8 -*-
#kaisya_data.py - 各会社のログイン情報を取得する
#引数は会社コードとする

#kaisya_dataの戻り値
#0：ログイン情報       #1：アドレス情報           #2：Excel情報
#0-0：会社名           #1-0：ログインアドレス     #2-0：Excel更新行
#0-1：店舗コード        #1-1：IPO取得アドレス     #2-1：Excelアドレス（未定）
#0-2：ユーザID
#0-3：パスワード
#0-4：取引パスワード

import os, yaml

#会社コードから情報を取得し、会社データとして返す
def kaisya_data(k_code, password_dir):
    ######## sbiログイン情報 ########
    with open(os.path.join(password_dir, 'sbi.yml')) as f:
        config = yaml.load(f)
        sbi_USER_ID = config['sbi_USER_ID']
        sbi_PASSWORD = config['sbi_PASSWORD']
    #################################

    #イベント管理エクセルの行番号を初期化
    row = 9

    k_dict = {#マスタ
              "0":[["master","","","",""],
                   ["","https://kakaku.com/stock/ipo/schedule/"],
                   [row - 7,"",""]],

              #楽天証券
              '1':[["rakuten","","","",""],
                   ["https://www.rakuten-sec.co.jp/","https://www.rakuten-sec.co.jp/web/domestic/ipo/"],
                   [row,"",""]],

              #ＳＢＩ証券
              '2':[["sbi","",sbi_USER_ID,sbi_PASSWORD,""],
                   ["https://www.sbisec.co.jp/ETGate","https://www.sbisec.co.jp/ETGate/?OutSide=on&_ControlID=WPLETmgR001Control&_DataStoreID=DSWPLETmgR001Control&burl=search_domestic&dir=ipo%2F&file=stock_info_ipo.html&cat1=domestic&cat2=ipo&getFlg=on&int_pr1=150313_cmn_gnavi:6_dmenu_04"],
                   [row + 1,"",""]],

              #ＳＭＢＣ日興証券
              '3':[["smbc","","","",""],
                   ["https://trade.smbcnikko.co.jp/Etc/1/webtoppage/","https://trade.smbcnikko.co.jp/Etc/1/webtoppage/"],
                   [row + 2,"",""]],

              #マネックス証券
              '4':[["monex","","","",""],
                   ["https://mst.monex.co.jp/pc/ITS/login/LoginIDPassword.jsp","https://www.monex.co.jp/"],
                   [row + 3,"",""]],

              #松井証券
              '5':[["matsui","","","",""],
                   ["https://www.deal.matsui.co.jp/ITS/login/MemberLogin.jsp","https://www.matsui.co.jp/market/stock/ipo/"],
                   [row + 4,"",""]],

              #ＧＭＯクリック証券
              '6':[["gmo","","","",""],
                   ["https://sec-sso.click-sec.com/loginweb/",""],
                   [row + 5,"",""]],

              #野村証券
              '7':[["nomura","","","",""],
                   ["https://hometrade.nomura.co.jp/web/rmfIndexWebAction.do","https://hometrade.nomura.co.jp/web/rmfIndexWebAction.do"],
                   [row + 6,"",""]],

              #ａｕカブコム証券
              '8':[["kabucom","","","",""],
                   ["https://s10.kabu.co.jp/_mem_bin/members/login.asp?/members/","https://kabu.com/item/ipo_po/meigara/default.html"],
                   [row + 7,"",""]],

              #大和証券
              '9':[["daiwa","","","",""],
                   ["https://www.daiwa.co.jp/PCC/HomeTrade/Account/m8301.html","https://www.daiwa.jp/products/equity/"],
                   [row + 8,"",""]],

              #東海東京証券
              '10':[["tokai","","","",""],
                    ["https://onlinetrade.tokaitokyo.co.jp/web/rmfIndexWebAction.do","http://www.tokaitokyo.co.jp/kantan/products/stock/ipo/index.html"],
                    [row + 9,"",""]],

              #岡三オンライン証券
              '11':[["okasan","","","",""],
                    ["https://trade.okasan-online.co.jp/web/","https://www.okasan-online.co.jp/jp/ipo/"],
                    [row + 10,"",""]],

              #ＤＭＭ．ｃｏｍ証券
              '12':[["dmm","","","",""],
                    ["https://trade.fx.dmm.com/comportal/Login.do?type=3","https://kabu.dmm.com/service/stock/ipo/"],
                    [row + 11,"",""]],

              #岩井コスモ証券
              '13':[["iwai","","","",""],
                    ["https://ic01.iwaicosmo.co.jp/Web/#/order/home/","https://ic01.iwaicosmo.co.jp/Web/#/order/home/"],
                    [row + 12,"",""]],

              #ライブスター証券
              '14':[["livestar","","","",""],
                    ["https://lv01.live-sec.co.jp/webbroker3/44/pc/WEB3AccountLogin.jsp","https://www.live-sec.co.jp/ipo/?gln"],
                    [row + 13,"",""]],

              #ストリーム
              '15':[["stream","","","",""],
                    ["",""],
                    [row + 14,"",""]],

              #みずほ証券
              '16':[["mizuho","","","",""],
                    ["https://netclub.mizuho-sc.com/mnc/login","https://info.www.mizuho-sc.com/ap/product/stock/ipo/index.html"],
                    [row + 15,"",""]],

              #ＬＩＮＥ証券
              '17':[["line","","","",""],
                    ["",""],
                    [row + 16,"",""]],

              #ネオモバイル証券
              '18':[["neomoba","","","",""],
                    ["https://trade.sbineomobile.co.jp/login","https://www.sbineomobile.co.jp/prd/dstock/ipo/"],
                    [row + 17,"",""]],

              #ＡＬＬ指定
              '99':[["","","","",""],
                    ["",""],
                    ["","",""]],
            }
    k_data = k_dict[k_code]
    return k_data



#kaisya_dataの戻り値
#0：ログイン情報       #1：アドレス情報           #2：Excel情報
#0-0：名称

#会社コードから情報を取得し、会社データとして返す
def bunbai_data(bun_code):
    #イベント管理エクセルの行番号を初期化
    row = 29

    b_dict = {#マスタ
              "0":[["bunbai","","","",""],
                   ["","http://立会外分売.jp/schedule/"],
                   [row,"",""]],

              #－－－－
              '1':[["","","","",""],
                   ["",""],
                   ["","",""]],

            }
    bun_data = b_dict[bun_code]
    return bun_data
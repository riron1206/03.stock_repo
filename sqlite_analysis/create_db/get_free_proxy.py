#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spys.one/en から公開プロキシのIPアドレスを取得する
https://github.com/GINK03/GIZYUTSUSHOTEN-08/blob/master/markdowns/060-5-proxies.md
Usage:
    $ activate stock
    $ python get_free_proxy.py  # カレントディレクトリにproxies.jsonができる。公開IPのリスト
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import re
from bs4 import BeautifulSoup
import json

# chromedriverのpathは適宜、自身の環境に合わせて編集すること
CHROMEDRIVER = r"C:\userApp\Selenium\chromedriver_win32\chromedriver.exe"

options = Options()
options.add_argument('--headless')
options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36")
options.add_argument(f"user-data-dir=/tmp/work")
options.add_argument('lang=ja')
options.add_argument("--no-sandbox")
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--window-size=1080,10800")
driver = webdriver.Chrome(options=options,
                          executable_path=CHROMEDRIVER)
                          # executable_path='/usr/bin/chromedriver')

proxies = set()
for proxy_src in ['http://spys.one/free-proxy-list/JP/', 'http://spys.one/en/free-proxy-list/']:
    driver.get('http://spys.one/en/free-proxy-list/')
    time.sleep(1.0)
    driver.find_element_by_xpath(
        "//select[@name='xpp']/option[@value='5']").click()
    time.sleep(1.0)
    html = driver.page_source
    soup = BeautifulSoup(html)
    [s.extract() for s in soup('script')]
    #print(soup.title.text)
    for tr in soup.find_all('tr'):
        if len(tr.find_all('td')) == 10:
            tds = tr.find_all('td')
            ip_port = tds[0].text.strip()
            protocol = re.sub(
                r'\(.*?\)', '', tds[1].text.strip()).lower().strip()
            proxy = f'{protocol}://{ip_port}'
            proxies.add(proxy)
proxies = list(proxies)

with open('proxies.json', 'w') as fp:
    json.dump(proxies, fp, indent=2)

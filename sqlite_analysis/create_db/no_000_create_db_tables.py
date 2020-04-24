#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBとテーブル作成
Usage:
    $ activate stock
    $ python no_000_create_db_tables.py
"""
import argparse

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
import util


def _create_table(sql, db_file_name):
    try:
        util.execute_sql(sql, db_file_name=db_file_name)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-db", "--db_file_name", type=str, default=r'D:\DB_Browser_for_SQLite\stock.db',
                    help="sqlite db file path.")
    args = vars(ap.parse_args())

    db_file_name = args['db_file_name']

    # brands:銘柄情報テーブル
    sql = """CREATE TABLE brands (
                code TEXT PRIMARY KEY, -- 銘柄コード
                name TEXT, -- 銘柄名（正式名称）
                short_name TEXT, -- 銘柄名（略称）
                market TEXT, -- 上場市場名
                sector TEXT, -- セクタ
                unit INTEGER -- 単元株数
             );"""
    _create_table(sql, db_file_name)

    # raw_prices:取得した生の株価テーブル。分割・併合情報を含めていない
    sql = """CREATE TABLE raw_prices (
                code TEXT, -- 銘柄コード
                date TEXT, -- 日付
                open REAL, -- 初値
                high REAL, -- 高値
                low REAL, -- 安値
                close REAL, -- 終値
                volume INTEGER, -- 出来高
                PRIMARY KEY(code, date)
             );"""
    _create_table(sql, db_file_name)

    # prices:株価テーブル。分割・併合情報を含めている
    sql = """CREATE TABLE prices (
                code TEXT, -- 銘柄コード
                date TEXT, -- 日付
                open REAL, -- 初値
                high REAL, -- 高値
                low REAL, -- 安値
                close REAL, -- 終値
                volume INTEGER, -- 出来高
                PRIMARY KEY(code, date)
             );"""
    _create_table(sql, db_file_name)

    # new_brands:上場情報テーブル
    sql = """CREATE TABLE new_brands ( -- 上場情報
                code TEXT, -- 銘柄コード
                date TEXT, -- 上場日
                PRIMARY KEY(code, date)
             );"""
    _create_table(sql, db_file_name)

    # delete_brands:廃止情報テーブル
    sql = """CREATE TABLE delete_brands ( -- 廃止情報
                code TEXT, -- 銘柄コード
                date TEXT, -- 廃止日
                PRIMARY KEY(code, date)
             );"""
    _create_table(sql, db_file_name)

    # divide_union_data:分割・併合情報テーブル
    sql = """CREATE TABLE divide_union_data (
                code TEXT, -- 銘柄コード
                date_of_right_allotment TEXT, -- 権利確定日
                before REAL, -- 分割・併合前株数の割合
                after REAL, -- 分割・併合後株数の割合
                PRIMARY KEY("code","date_of_right_allotment")
             );"""
    _create_table(sql, db_file_name)

    # applied_divide_union_data:pricesテーブルに反映させた分割・併合情報テーブル
    sql = """CREATE TABLE applied_divide_union_data (
                code TEXT, -- 銘柄コード
                date_of_right_allotment TEXT, -- 権利確定日
                PRIMARY KEY(code, date_of_right_allotment)
             );"""
    _create_table(sql, db_file_name)

    # raw_ratings:生の目標株価（レーティング）情報テーブル
    sql = """CREATE TABLE raw_ratings (
               date TEXT, -- 公開日
               code TEXT, -- 銘柄コード
               think_tank TEXT, -- 目標株価を公表した証券会社などの名前
               rating TEXT, -- レーティング
               target REAL, -- 目標株価（未調整）
               PRIMARY KEY(date, code, think_tank)
             );"""
    _create_table(sql, db_file_name)

    # ratings:目標株価（レーティング）情報テーブル
    # raw_ratingsとratingsの違いは、targetに格納された目標株価
    # ratingsは株式分割・併合情報をもとに調整後株価が入る
    sql = """CREATE TABLE ratings (
               date TEXT, -- 公開日
               code TEXT, -- 銘柄コード
               think_tank TEXT, -- 目標株価を公表した証券会社などの名前
               rating TEXT, -- レーティング
               target REAL, -- 目標株価（調整後）
               PRIMARY KEY(date, code, think_tank)
             );"""
    _create_table(sql, db_file_name)

    # quarterly_results: 営業利益を含む四半期ごとの決算情報
    sql = """CREATE TABLE quarterly_results (
                code TEXT, -- 銘柄コード
                term TEXT, -- 決算期 （例: 2018年4～6月期ならば2018-06）
                date TEXT, -- 決算発表日
                sales INTEGER, -- 売上高（単位:百万円）
                op_income INTEGER, -- 営業利益（単位:百万円）
                ord_income INTEGER, -- 経常利益（単位:百万円）
                net_income INTEGER -- 最終利益（単位:百万円）
             );"""
    _create_table(sql, db_file_name)
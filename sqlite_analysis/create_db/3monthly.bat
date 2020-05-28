@rem 作成日2020/04/24 DB3ヵ月ごとに更新

set MY_DB=D:\DB_Browser_for_SQLite\stock.db
set CSV_DIR=D:\DB_Browser_for_SQLite\csvs\kabuoji3
set CSV_DIR2=D:\DB_Browser_for_SQLite\csvs\traders
set CSV_DIR3=D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly


call activate stock

@rem 上場・廃止銘柄スクレイピング + 上場テーブルと廃止テーブルに銘柄追加 + 銘柄情報テーブルに上場銘柄追加（廃止銘柄は消さずに残す）
@rem これは3か月ごとの実行でいいと思う
call python no_030_get_new_delete_brands_info.py -db %MY_DB%
echo %date% %time% no_030_get_new_delete_brands_info.py end. >> log.txt

@rem 全テーブル更新する場合はdaily.bat呼び出す
call daily.bat

@rem 営業利益を含む四半期ごとの決算情報スクレイピング + テーブルに情報追加。-uで最新レコードのみ追加する
@rem これは3か月ごとの実行でいい
call python no_070_quarterly_results.py -db %MY_DB% -dir %CSV_DIR3% -b_dir %CSV_DIR% -u

@rem 空行1行追加
echo. >> log.txt
echo %date% %time% #### 3monthly update all end!!! #### >> log.txt 

pause
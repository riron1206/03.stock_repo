@rem 作成日2020/04/21 DB日時更新

set MY_DB=D:\DB_Browser_for_SQLite\stock.db
set CSV_DIR=D:\DB_Browser_for_SQLite\csvs\kabuoji3
set CSV_DIR2=D:\DB_Browser_for_SQLite\csvs\traders
set CSV_DIR3=D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly


call activate stock

@rem 上場・廃止銘柄スクレイピング + 上場テーブルと廃止テーブルに銘柄追加 + 銘柄情報テーブルに上場銘柄追加（廃止銘柄は消さずに残す）
call python no_030_get_new_delete_brands_info.py -db %MY_DB%

@rem 最新株価スクレイピング + 新規レコードcsvに追加 + 株価テーブルに株価追加。-uで最新株価のみ追加する。1銘柄1件づつ追加する場合1時間ぐらいかかる
call python no_021_csv_to_db.py  -db %MY_DB% -dir %CSV_DIR% -u

@rem 株価csvから分割・併合テーブルに分割・併合情報追加
call python no_040_csv_to_divide_union_data.py -db %MY_DB% -dir %CSV_DIR%

@rem 今月のレーティング情報スクレイピング + レーティングテーブルに情報追加
call python no_050_rating_data.py -db %MY_DB% -dir %CSV_DIR2%

@rem 分割・併合テーブルの内容を株価テーブルとレーティングテーブルに反映
call python no_060_apply_divide_union_data.py -db %MY_DB%

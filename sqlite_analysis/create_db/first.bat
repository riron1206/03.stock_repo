@rem 作成日2020/04/22 DB構築

set MY_DB=D:\DB_Browser_for_SQLite\stock.db
set CSV_DIR=D:\DB_Browser_for_SQLite\csvs\kabuoji3
set CSV_DIR2=D:\DB_Browser_for_SQLite\csvs\traders
set CSV_DIR3=D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly


call activate stock

@rem DB + テーブル作成
call python no_000_create_db_tables.py -db %MY_DB%

@rem 銘柄情報スクレイピング + 銘柄情報テーブルに全銘柄追加。3時間ぐらいかかる
call python no_010_get_brands_info.py -db %MY_DB%

@rem スクレイピングで株価csv全銘柄ダウンロード。30時間以上かかった
call python no_020_kabuoji3_csv_download.py -o %CSV_DIR%

@rem 株価テーブルに全株価csv追加
call python no_021_csv_to_db.py -db %MY_DB% -dir %CSV_DIR%

@rem 上場・廃止銘柄スクレイピング + 上場テーブルと廃止テーブルに銘柄追加 + 銘柄情報テーブルに上場銘柄追加
call python no_030_get_new_delete_brands_info.py -db %MY_DB%

@rem 株価csvから分割・併合テーブルに分割・併合情報追加
call python no_040_csv_to_divide_union_data.py -db %MY_DB% -dir %CSV_DIR%

@rem 古いレーティング情報スクレイピング + レーティングテーブルに情報追加
call python no_050_rating_data.py -db %MY_DB% -dir %CSV_DIR2% --is_old
@rem 今月のレーティング情報スクレイピング + レーティングテーブルに情報追加
call python no_050_rating_data.py -db %MY_DB% -dir %CSV_DIR2%

@rem 分割・併合テーブルの内容を株価テーブルとレーティングテーブルに反映。
call python no_060_apply_divide_union_data.py -db %MY_DB%

@rem 営業利益を含む四半期ごとの決算情報スクレイピング + テーブルに情報追加。1時間ぐらいかかる
call python no_070_quarterly_results.py -db %MY_DB% -dir %CSV_DIR3% -b_dir %CSV_DIR%


@rem ログファイル作成
echo %date% %time% #### first create all end!!! #### > log.txt

pause
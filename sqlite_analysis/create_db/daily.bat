@rem 作成日2020/04/21 DB日時更新

set MY_DB=D:\DB_Browser_for_SQLite\stock.db
set CSV_DIR=D:\DB_Browser_for_SQLite\csvs\kabuoji3
set CSV_DIR2=D:\DB_Browser_for_SQLite\csvs\traders
set CSV_DIR3=D:\DB_Browser_for_SQLite\csvs\kabutan_quarterly


call activate stock

@rem 空行1行追加
echo. >> log.txt
echo %date% %time% #### daily update start!!! #### >> log.txt


@rem 公開プロキシのIPアドレスを取得。公開プロキシ介すると遅いし失敗することあるのでやめておく
@rem call python get_free_proxy.py
@rem echo %date% %time% get_free_proxy.py end. >> log.txt


@rem 最新株価スクレイピング + 新規レコードcsvに追加 + 株価テーブルに株価追加。-uで最新株価のみ追加する。1銘柄1件づつ追加する場合1時間ぐらいかかる
call python no_021_csv_to_db.py  -db %MY_DB% -dir %CSV_DIR% -u
echo %date% %time% no_021_csv_to_db.py end. >> log.txt


@rem 株価csvから分割・併合テーブルに分割・併合情報追加
call python no_040_csv_to_divide_union_data.py -db %MY_DB% -dir %CSV_DIR%
echo %date% %time% no_040_csv_to_divide_union_data.py end. >> log.txt


@rem 今月のレーティング情報スクレイピング + レーティングテーブルに情報追加
call python no_050_rating_data.py -db %MY_DB% -dir %CSV_DIR2%
echo %date% %time% no_050_rating_data.py end. >> log.txt 


@rem 分割・併合テーブルの内容を株価テーブルとレーティングテーブルに反映
call python no_060_apply_divide_union_data.py -db %MY_DB%
echo %date% %time% no_060_apply_divide_union_data.py end. >> log.txt 


echo %date% %time% #### daily update all end!!! #### >> log.txt 

@rem pause
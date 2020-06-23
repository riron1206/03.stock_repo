@rem 作成日2020/04/23 DB日時更新bat実行して、終了したらPC落とす

@rem daily.bat呼び出す
cd C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\sqlite_analysis\create_db
call daily.bat

@rem order.csv作成
cd C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\sqlite_analysis\auto_trade
call 01.run_make_order_csv.bat

@rem シャットダウンコマンド メッセージを出力し60秒後にパソコンを閉じます
shutdown /s /t 60 /c "後60秒でシャットダウンします。"


@rem 作成日2020/04/23 DB日時更新bat実行して、終了したらPC落とす

@rem daily.bat呼び出す
call daily.bat

@rem シャットダウンコマンド メッセージを出力し60秒後にパソコンを閉じます
shutdown /s /t 60 /c "後60秒でシャットダウンします。"


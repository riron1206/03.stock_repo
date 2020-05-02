@rem 作成日2020/05/02 セレニウムでsbiログインして注文する。注文成功したらinputディレクトリのcsvファイルに実行日が入るらしい

call activate stock

echo. >> log.txt
echo %date% %time% #### auto_trade_sbi.py start!!! #### >> log.txt

call python auto_trade_sbi.py >> log.txt

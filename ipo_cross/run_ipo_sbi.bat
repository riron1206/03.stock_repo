@rem 作成日2020/4/17 sbiでipo登録を実行する

timeout 60

call activate stock

cd ipo_cross

call python main.py -o output -k_codes 2 -p C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\ipo_cross\password

pause
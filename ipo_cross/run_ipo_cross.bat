@rem 作成日2020/3/29 クロス取引とipo登録を実行する

call activate stock

cd ipo_cross

call python main.py -o output -i C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\ipo_cross\input -p C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\ipo_cross\password

pause

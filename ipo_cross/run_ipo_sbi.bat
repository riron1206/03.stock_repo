@rem �쐬��2020/4/17 sbi��ipo�o�^�����s����

timeout 60

call conda activate stock

cd ipo_cross

call python main.py -o output -k_codes 2 -p C:\Users\shingo\jupyter_notebook\stock_work\03.stock_repo\ipo_cross\password

pause

@rem 作成日2020/05/03 シグナルcsvファイルを更新して銘柄コードをSlackに送る

call activate stock

echo. >> log.txt
echo %date% %time% #### make_order_csv.py start!!! #### >> log.txt

call python make_order_csv.py -all

call python post_my_slack.py
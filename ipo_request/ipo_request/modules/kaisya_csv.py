#! python3
# -*- coding: utf-8 -*-
#kaisya_csv.py

#各会社のIPOリストからCSVファイルを作成する
#引数はIPOリストと会社データとする

#使用するモジュールのインポート
import os

#マスタリストからCSVファイルを作成する。
def master_csv(ipo_list,k_data,output_dir):
    with open(os.path.join(output_dir, f"{str(k_data[0][0])}.csv"), "w") as f:
        for d in ipo_list[::-1]:
            f.write("{},{},{},{}\n".format(d[0],d[1],d[2],d[3]))



#会社リストからCSVファイルを作成する。
def kaisya_csv(ipo_list,k_data,output_dir):
    with open(os.path.join(output_dir, f"{str(k_data[0][0])}.csv"), "w") as f:
        for d in ipo_list:
            f.write("{},{}\n".format(d[0],d[1]))



#分売リストからCSVファイルを作成する。
def bunbai_csv(bun_list,bun_data,output_dir):
    with open(os.path.join(output_dir, f"{str(bun_data[0][0])}_list.csv"), "w") as f:
        for d in bun_list:
            f.write("{},{},{},{}\n".format(d[0],d[1],d[2],d[3]))
# -*- coding: utf-8 -*-
"""
ペアトレードに向いている銘柄を探索
※ペアトレード：安な銘柄を「買い」、割高な銘柄を「空売り」しておき、株価が近づいたタイミングで反対売買を行い利益を確定する

<具体的な処理>
1. 1銘柄の累積収益率計算
2. 2銘柄の累積収益率の残差に対してADF検定を実施し、p値の距離行列を作成
3. p値の距離行列からクラスタリング
4. p値<0.1のペアについて収益率など可視化
- p値の距離が近い銘柄同士は共和分(=非定常のデータの足し算は定常過程になる性質)の関係にあると解釈
- 共和分により、その2銘柄の収益率の累積積の差（残差）は0を基準に行ったり来たりするはずなので。ペアトレードに向いている銘柄といえる

※ADF検定は単位根過程(=非定常だけど、差分をとると（弱）定常になるようなデータ)かどうかの検定
　p値<0.1なら単位根過程でないと解釈する

2銘柄の累積収益率の残差グラフに周期性があるか確認する必要があるためnotebookで実行すべき
参考:run_pair_trade.ipynb

予測のみ実行する場合は以下のコマンドで実行できる
    $ activate stock
    $ python pair_trade.py -c1 7202 -c2 7269  # 7202,7269について累積収益率の残差グラフ作成
"""
import argparse
import datetime
import os
import joblib
import pathlib
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # グラフの体裁きれいにする
plt.rcParams["font.family"] = 'Yu Gothic'   # Yu Gothic指定すれば日本語出せる


def plot_dendrogram(distance_matrix: np.ndarray, names: list, codes: list, method='single', figsize=(15, 8), output_dir=None):
    """ 距離行列から階層型クラスタリングで樹形図（デンドログラム）をplotする """
    # 距離行列からクラスタリングできるように変換
    d_squareform = squareform(distance_matrix)

    # 階層型クラスタリング実行 クラスタリング手法は methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]　ある
    z = linkage(d_squareform, method='single', metric='euclidean')

    # linkage関数は、クラスタリングのステップ毎に [クラスター番号1, クラスター番号2, クラスタリングに伴う距離の変化, 新しいクラスターに属するデータ数] を返す
    df_z = pd.DataFrame(z, columns=['index1', 'index2', 'distance', 'date_count'])
    df_z['index1_name'] = df_z['index1'].apply(lambda x: f'{names[int(x)]}:{codes[int(x)]}' if int(x) < len(names) else np.nan)
    df_z['index2_name'] = df_z['index2'].apply(lambda x: f'{names[int(x)]}:{codes[int(x)]}' if int(x) < len(names) else np.nan)

    labels = [n + ':' + str(c) for n, c in zip(names, codes)]

    # デンドログラムを描く
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(z, labels=labels, ax=ax)
    plt.title(f'2銘柄の累積収益率の残差のp値について:{method}')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, method + '_dendro.png'), bbox_inches="tight")
    plt.show()

    return df_z


def calc_return_rate_cumprod(csv_path: str, train_start_date=None):
    """ 1銘柄の株価csvから収益率の累積積を計算 """
    # 1銘柄の株価csvロード
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0], dtype='float', encoding='shift-jis')
    # print(df)
    # print(csv_path)

    # 期間指定する
    df = df if train_start_date is None else df[train_start_date:]

    # 収益率計算
    df['return_rate'] = (df['終値調整'] / df['終値調整'].shift(1))

    # 累積積を計算
    df_return_rate = pd.DataFrame({str(pathlib.Path(csv_path).stem): df['return_rate']})
    df['return_rate_cumprod'] = df_return_rate.apply(np.cumprod)

    return df


def cluster_adf_p_val(data_dir: str, codes: list, names: list, train_start_date=None, figsize=(15, 8), output_dir=None):
    """ calc_return_rate_cumprod(), plot_dendrogram() 一気に実行 """
    # 1銘柄の株価csvから収益率の累積積を計算
    dict_df_csv = {}
    dict_df_return_rate_cumprod = {}
    for code in codes:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        df_csv = calc_return_rate_cumprod(csv_path, train_start_date=train_start_date)
        dict_df_csv[str(code)] = df_csv
        dict_df_return_rate_cumprod[str(code)] = df_csv['return_rate_cumprod'].dropna()

    # 2銘柄の累積収益率の残差でADF検定のp値計算して距離行列作成
    distance_matrix = np.zeros((len(codes), len(codes)))
    for i, code1 in tqdm(enumerate(codes)):
        for j, code2 in enumerate(codes):
            ts1 = dict_df_return_rate_cumprod[str(code1)]
            ts2 = dict_df_return_rate_cumprod[str(code2)]
            ts_diff = (ts1 - ts2).dropna()
            p_val = sm.tsa.stattools.adfuller(ts_diff, autolag='AIC')[1]  # ADF検定
            distance_matrix[i, j] = p_val
    distance_matrix = np.nan_to_num(distance_matrix, nan=0.0)  # 欠損値を0に置換

    # ADF検定のp値の距離行列でクラスタリング
    df_z = plot_dendrogram(distance_matrix, names, codes)

    return dict_df_csv, df_z


def plot_pair_return_rate(data_dir, code1, code2, start_date='2020-01-01', figsize=(15, 8), output_dir=None):
    """ 2銘柄の収益率などplotする """
    csv_path1 = os.path.join(data_dir, f'{code1}.csv')
    csv_path2 = os.path.join(data_dir, f'{code2}.csv')
    df1 = pd.read_csv(csv_path1, index_col=0, parse_dates=[0], dtype='float', encoding='shift-jis')
    df2 = pd.read_csv(csv_path2, index_col=0, parse_dates=[0], dtype='float', encoding='shift-jis')
    df1 = df1[start_date:]
    df2 = df2[start_date:]

    # 収益率の残差
    df1['return_rate'] = df1['終値調整'] / df1['終値調整'].shift(1)
    df2['return_rate'] = df2['終値調整'] / df2['終値調整'].shift(1)
    return_rate_resid = df1['return_rate'] - df2['return_rate']

    # 累積収益率の残差
    df1['return_rate_cumprod'] = df1[['return_rate']].apply(np.cumprod)
    df2['return_rate_cumprod'] = df2[['return_rate']].apply(np.cumprod)
    return_rate_resid_cumprod = df1['return_rate_cumprod'] - df2['return_rate_cumprod']

    df1 = df1.rename(columns={'終値調整': f'{code1}_終値調整', 'return_rate': f'{code1}_return_rate'})
    df2 = df2.rename(columns={'終値調整': f'{code2}_終値調整', 'return_rate': f'{code2}_return_rate'})

    # 終値調整plot
    fig, ax = plt.subplots(figsize=figsize)
    df1[[f'{code1}_終値調整']].plot(marker='x', ax=ax)
    df2[[f'{code2}_終値調整']].plot(marker='x', ax=ax)
    plt.title('2銘柄の終値調整')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{code1}_{code2}_close.png'), bbox_inches="tight")
    plt.show()

    # # 収益率plot
    # fig, ax = plt.subplots(figsize=figsize)
    # df1[[f'{code1}_return_rate']].plot(marker='x', ax=ax)
    # df2[[f'{code2}_return_rate']].plot(marker='x', ax=ax)
    # plt.title('2銘柄の収益率')
    # if output_dir is not None:
    #     plt.savefig(os.path.join(output_dir, f'{code1}_{code2}_return_rate.png'), bbox_inches="tight")
    # plt.show()

    # # 収益率の残差plot
    # fig, ax = plt.subplots(figsize=figsize)
    # return_rate_resid.plot(marker='x', ax=ax)
    # plt.title('2銘柄の収益率の残差')
    # if output_dir is not None:
    #     plt.savefig(os.path.join(output_dir, f'{code1}_{code2}_return_rate_resid.png'), bbox_inches="tight")
    # plt.show()

    # 累積収益率の残差plot
    fig, ax = plt.subplots(figsize=figsize)
    return_rate_resid_cumprod.plot(marker='x', ax=ax)
    plt.title('2銘柄の累積収益率の残差')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{code1}_{code2}_return_rate_resid_cumprod.png'), bbox_inches="tight")
    plt.show()


def search_pair_trade(data_dir, codes, names, output_dir, p_threshold=0.05, train_start_date='2010-01-01'):
    """ ペアトレードに向いている銘柄を探索 """
    # calc_return_rate_cumprod(), plot_dendrogram() 一気に実行
    dict_df_csv, df_z = cluster_adf_p_val(data_dir, codes, names, train_start_date=train_start_date, output_dir=output_dir)

    # ペアのクラスター + p値が小さいペア確認
    df_pair = df_z[(df_z['date_count'] == 2.0) & (df_z['distance'] < p_threshold)]

    # ペアの収益率など可視化
    for ind, row in df_pair.iterrows():
        code1 = row['index1_name'].split(':')[1]
        code2 = row['index2_name'].split(':')[1]
        print(f"---- {row['index1_name']}, {row['index2_name']} -----")
        plot_pair_return_rate(data_dir, code1, code2, start_date='2020-01-01', output_dir=output_dir)


def test_func():
    """
    テスト駆動開発でのテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s pair_trade.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行。-s付けるとprint()の内容出してくれる
    """
    matplotlib.use('Agg')  # GUI使えない場合はこれつける

    def _test_search_pair_trade():
        """ search_pair_trade() のテスト """
        # テストデータ
        output_dir = r'D:\work\02_keras_py\experiment\01_code_test\output_test\tmp'
        data_dir = r'D:\DB_Browser_for_SQLite\csvs\kabuoji3'
        automotive_names = ['イーソル', '日産自', 'いすゞ', 'トヨタ', '日野自', '三菱自', 'マツダ', 'ホンダ', 'スズキ', 'ＳＵＢＡＲＵ']
        automotive_codes = [4420, 7201, 7202, 7203, 7205, 7211, 7261, 7267, 7269, 7270]

        # ペアトレードに向いている銘柄を探索
        search_pair_trade(data_dir, automotive_codes, automotive_names, output_dir, p_threshold=0.05)
    _test_search_pair_trade()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", default=r"output", help="output dir path")
    ap.add_argument("-d", "--data_dir", default=r"D:\DB_Browser_for_SQLite\csvs\kabuoji3", help="stock csv dir path")
    ap.add_argument("-c1", "--pred_stock_code1", type=int, default=7201, help="predict stock code1")
    ap.add_argument("-c2", "--pred_stock_code2", type=int, default=7202, help="predict stock code2")
    ap.add_argument("-p_s", "--pred_start_date", type=str, default=None, help="predict start date")
    return vars(ap.parse_args())


def plot_pair_trade():
    """ 指定した2銘柄でグラフ可視化のみ実行する """
    args = get_args()
    os.makedirs(args['output_dir'], exist_ok=True)

    # 予測結果の表示開始日
    now = datetime.datetime.now()  # date型にする場合は datetime.date.today()
    pred_start = now - datetime.timedelta(days=20) if args['pred_start_date'] is None else args['pred_start_date']
    pred_start = pred_start.strftime('%Y-%m-%d')  # => '2019-08-02'

    # 2銘柄でグラフ可視化
    plot_pair_return_rate(args['data_dir'], args['pred_stock_code1'], args['pred_stock_code2'],
                          start_date=pred_start, output_dir=args['output_dir'])


if __name__ == '__main__':
    matplotlib.use('Agg')  # GUI使えない場合はこれつける
    plot_pair_trade()  # 2銘柄でグラフ可視化

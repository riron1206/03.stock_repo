# -*- coding: utf-8 -*-
"""
ペアトレードに向いている銘柄を探索
※ペアトレード：安な銘柄を「買い」、割高な銘柄を「空売り」しておき、株価が近づいたタイミングで反対売買を行い利益を確定する

<具体的な処理>
1. 各銘柄の累積収益率計算
2. 2銘柄について、一方の累積収益率を説明変数としてもう一方の累積収益率を予測する線形回帰モデル作成
3. 線形回帰モデルの予測と正解ラベル値の残差に対してADF検定を実施し、p値の距離行列を作成
4. p値の距離行列からクラスタリング
5. p値<0.1のペアについて収益率など可視化
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


class Model():

    def make_lr_model(self, merge_df_return_rate, code1: str, code2: str, figsize=(15, 8), output_dir=None, is_plot=True):
        """ 銘柄の累積収益率について、線形回帰モデル作成 """
        _df_return_rate = merge_df_return_rate[[code1, code2]].dropna()
        X = _df_return_rate[code1]  # 特徴量(1銘柄の累積収益率)
        Y = _df_return_rate[code2]  # 目的変数（1銘柄の累積収益率）

        # 一応トレーニング・テストデータ分割
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

        # ValueError: Expected 2D array, got 1D array instead:対策 https://qiita.com/nkay/items/8cd3c3635fe39ecb92e0
        _X_train = X_train.values.reshape(-1, 1)
        _X_test = X_test.values.reshape(-1, 1)
        _Y_train = Y_train.values.reshape(-1, 1)
        _Y_test = Y_test.values.reshape(-1, 1)
        # print(X.shape, Y.shape)

        # LinearRegression  y = (1)*x1 + (-1)*x2
        model = LinearRegression()
        model.fit(_X_train, _Y_train)

        # 予測　
        Y_pred = model.predict(_X_test)
        Y_pred = Y_pred.flatten()  # 1次元にする

        # # 平均絶対誤差(MAE) 正解値と予測値の差分の絶対値の平均
        # mae = mean_absolute_error(_Y_test, Y_pred)
        # # 平方根平均二乗誤差（RMSE）正解値と予測値の差分の二乗を平均し、平方したもの
        # rmse = np.sqrt(mean_squared_error(_Y_test, Y_pred))
        # # スコア
        # score = model.score(_X_test, _Y_test)
        # print("MAE = %.2f,  RMSE = %.2f,  score = %.2f" % (mae, rmse, score))
        # print("Coef:重み = ", model.coef_)
        # print("Intercept:バイアス項 =", model.intercept_)

        df_pred = pd.DataFrame({'Y_pred': Y_pred, 'date': Y_test.index})
        df_pred['date'] = pd.to_datetime(df_pred['date'], format='%Y-%m-%d')

        if is_plot:
            fig, ax = plt.subplots(figsize=figsize)  # subplotでfigsize指定
            Y.plot(ax=ax)
            df_pred.plot(x='date', y='Y_pred', marker='x', ax=ax)
            plt.title(f'{code2}の累積収益率の予測（説明変数は{code1}）')
            plt.legend()
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, 'lr_test.png'), bbox_inches="tight")
            plt.show()

        return model

    def save_model_info(self, model, training_info: dict, preprocess_pipeline=None, output_path='.'):
        """
        モデルの前処理や使うファイル、ハイパーパラメータなどの情報を保存
        ※「このモデルはどうやって学習されたんや！？」
        　「このモデルはどんな性能なんや！？」
        「このモデルにデータ投入するための、学習済みの前処理パイプがない！！」
        みたいな事態にならないようにデータ残す
        https://qiita.com/sugulu/items/c0e8a5e6b177bfe05e99
        Usage:
            test_func()の_test_save_model_info() 参照
        """
        from datetime import datetime, timedelta, timezone
        import joblib

        JST = timezone(timedelta(hours=+9), "JST")  # 日本時刻に
        now = datetime.now(JST).strftime("%Y%m%d_%H%M%S")  # 現在時間を取得

        # 出力先がディレクトリならファイル名に現在時刻付ける
        filepath = os.path.join(output_path, "model_info_" + now + ".joblib") if os.path.isdir(output_path) else output_path

        # 学習データ、モデルの種類、ハイパーパラメータの情報に現在時刻も詰める
        training_info["save_date"] = now

        save_data = {
            "preprocess_pipeline": preprocess_pipeline,
            "trained_model": model,
            "training_info": training_info}

        # 保存
        joblib.dump(save_data, filepath)
        # print("INFO: save file. {}".format(filepath))
        return save_data, filepath

    def predict_load_model_residual(self, model_path: str, X_test, Y_test, out_png=None):
        """ 作成したモデルファイルロードしてpredictし、正解ラベルとの差(残差)を出す """
        model_info = joblib.load(model_path)

        # ValueError: Expected 2D array, got 1D array instead:対策 https://qiita.com/nkay/items/8cd3c3635fe39ecb92e0
        _X_test = X_test.values.reshape(-1, 1)

        Y_pred = model_info['trained_model'].predict(_X_test)
        Y_pred = Y_pred.flatten()  # 1次元にする

        _Y_test = Y_test.values.flatten()  # 1次元にする
        # 残差
        residual = _Y_test - Y_pred

        df_resid = pd.DataFrame({'residual': residual, 'date': X_test.index})
        df_resid['date'] = pd.to_datetime(df_resid['date'], format='%Y-%m-%d')

        # 残差のグラフ表示
        if out_png is not None:
            fig, ax = plt.subplots(figsize=(15, 8))  # subplotでfigsize指定
            df_resid.plot(x='date', y='residual', marker='x', ax=ax)
            plt.title(f'{str(Y_test.name)}の累積収益率の正解ラベルとの差(残差)')
            if out_png is not None:
                plt.savefig(out_png, bbox_inches="tight")
            plt.show()

        return df_resid

    def run_pair_make_residual_lr_model(self, merge_df_return_rate, c1: str, c2: str, output_dir: str, pred_start_date: str):
        """ 銘柄のペアを変えてモデル作成 """
        model = self.make_lr_model(merge_df_return_rate, c1, c2, output_dir=output_dir, is_plot=False)

        # モデル保存
        training_info = {'training_data': f'{c1}.csv', 'Y': f'{c2}.csv', 'model_type': 'LinearRegression', 'hyper_pram': 'default'}
        save_data, filepath = self.save_model_info(model,
                                                   training_info,
                                                   preprocess_pipeline=None,
                                                   output_path=os.path.join(output_dir, f'lr_{c2}_by_{c1}.joblib'))

        # モデルロードして予測
        _df_return_rate = merge_df_return_rate[[c1, c2]].dropna()
        X_test = _df_return_rate.loc[pred_start_date:][c1]
        Y_test = _df_return_rate.loc[pred_start_date:][c2]
        df_resid = self.predict_load_model_residual(filepath, X_test, Y_test)
        return df_resid


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


def cluster_adf_p_val(data_dir: str, codes: list, names: list, p_threshold=0.05, train_start_date=None, figsize=(15, 8), output_dir=None):
    """ calc_return_rate_cumprod(), plot_dendrogram() 一気に実行 """
    # 1銘柄の株価csvから収益率の累積積を計算
    dict_df_csv = {}
    dict_ts_return_rate_cumprod = {}
    for code in codes:
        csv_path = os.path.join(data_dir, f'{code}.csv')
        df_csv = calc_return_rate_cumprod(csv_path, train_start_date=train_start_date)
        dict_df_csv[str(code)] = df_csv
        dict_ts_return_rate_cumprod[str(code)] = df_csv['return_rate_cumprod'].dropna()

    # 2銘柄の累積収益率の残差でADF検定のp値計算して距離行列作成
    distance_matrix = np.zeros((len(codes), len(codes)))
    for i, code1 in tqdm(enumerate(codes)):
        # 1つ目の銘柄
        ts1 = dict_ts_return_rate_cumprod[str(code1)]
        # p_val_ts1 = sm.tsa.stattools.adfuller(ts1, autolag='AIC')[1]  # ADF検定

        for j, code2 in zip(range(i + 1, len(codes)), codes[i + 1:]):
            # 2つ目の銘柄
            ts2 = dict_ts_return_rate_cumprod[str(code2)]
            # p_val_ts2 = sm.tsa.stattools.adfuller(ts2, autolag='AIC')[1]  # ADF検定

            # 単位根過程を保証するために、1銘柄単体でADF検定のp値 > しきい値の銘柄である必要がある
            # if (code1 != code2) and (p_val_ts1 > p_threshold) and (p_val_ts2 > p_threshold):  # 距離行列0ばかりになり、樹形図書けないのでやめた
            if (code1 != code2):
                # print(f'---- {code1}, {code2} ----')
                df1 = dict_df_csv[str(code1)][['return_rate_cumprod']].rename(columns={'return_rate_cumprod': str(code1)})
                df2 = dict_df_csv[str(code2)][['return_rate_cumprod']].rename(columns={'return_rate_cumprod': str(code2)})
                merge_df_return_rate = df1.join(df2, how='outer')

                # モデル作成+予測して、正解との残差出す
                df_resid = Model().run_pair_make_residual_lr_model(merge_df_return_rate, str(code1), str(code2), output_dir, train_start_date)

                # モデルとの残差について、ADF検定
                p_val = sm.tsa.stattools.adfuller(df_resid['residual'], autolag='AIC')[1]
                # print(p_val)

                # 距離行列にp値詰める
                distance_matrix[i, j] = p_val
                distance_matrix[j, i] = p_val  # 対角成分
            else:
                distance_matrix[i, j] = 0.0
    # print(distance_matrix)
    # ADF検定のp値の距離行列でクラスタリング
    df_z = plot_dendrogram(distance_matrix, names, codes)

    return dict_df_csv, df_z


def plot_pair_residual(data_dir, code1, code2, output_dir, start_date='2020-01-01', figsize=(15, 8)):
    """ 2銘柄の累積収益率の残差などplotする """
    csv_path1 = os.path.join(data_dir, f'{code1}.csv')
    csv_path2 = os.path.join(data_dir, f'{code2}.csv')
    df1 = pd.read_csv(csv_path1, index_col=0, parse_dates=[0], dtype='float', encoding='shift-jis')
    df2 = pd.read_csv(csv_path2, index_col=0, parse_dates=[0], dtype='float', encoding='shift-jis')
    df1 = df1[start_date:]
    df2 = df2[start_date:]

    # 収益率
    df1['return_rate'] = df1['終値調整'] / df1['終値調整'].shift(1)
    df2['return_rate'] = df2['終値調整'] / df2['終値調整'].shift(1)
    # 累積収益率
    df1[str(code1)] = df1[['return_rate']].apply(np.cumprod)
    df2[str(code2)] = df2[['return_rate']].apply(np.cumprod)

    # 列名変更
    df1 = df1.rename(columns={'終値調整': f'{code1}_終値調整'})
    df2 = df2.rename(columns={'終値調整': f'{code2}_終値調整'})

    # 終値調整plot
    fig, ax = plt.subplots(figsize=figsize)
    df1[[f'{code1}_終値調整']].plot(marker='x', ax=ax)
    df2[[f'{code2}_終値調整']].plot(marker='x', ax=ax)
    plt.title('2銘柄の終値調整')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{code1}_{code2}_close.png'), bbox_inches="tight")
    plt.show()

    # モデルの入力データ
    df1 = df1[[str(code1)]]
    df2 = df2[[str(code2)]]
    merge_df_return_rate = df1.join(df2, how='outer')
    _df_return_rate = merge_df_return_rate.dropna()
    X_test = _df_return_rate[str(code1)]
    Y_test = _df_return_rate[str(code2)]

    # モデルファイル
    model_path = os.path.join(output_dir, f'lr_{str(code2)}_by_{str(code1)}.joblib')
    out_png = os.path.join(output_dir, f'lr_{code2}_by_{code1}.png')
    if os.path.exists(model_path) == False:  # 距離行列は対角成分しか計算できないので
        model_path = os.path.join(output_dir, f'lr_{str(code1)}_by_{str(code2)}.joblib')
        out_png = os.path.join(output_dir, f'lr_{code1}_by_{code2}.png')
        X_test = _df_return_rate[str(code2)]
        Y_test = _df_return_rate[str(code1)]

    # モデルとの残差計算+plot
    _ = Model().predict_load_model_residual(model_path, X_test, Y_test, out_png=out_png)


def search_pair_trade(data_dir, codes, names, output_dir, p_threshold=0.05, train_start_date='2010-01-01'):
    """ ペアトレードに向いている銘柄を探索 """
    # calc_return_rate_cumprod(), plot_dendrogram() 一気に実行
    dict_df_csv, df_z = cluster_adf_p_val(data_dir, codes, names,
                                          p_threshold=p_threshold, train_start_date=train_start_date, output_dir=output_dir)
    # print(df_z)
    # ペアのクラスター + p値が小さいペア確認
    df_pair = df_z[(df_z['date_count'] == 2.0) & (df_z['distance'] < p_threshold)]
    print('df_pair:\n', df_pair)

    # ペアの収益率など可視化
    for ind, row in df_pair.iterrows():
        # print(f"{row['index1_name']}, {row['index2_name']}")
        code1 = row['index1_name'].split(':')[1]
        code2 = row['index2_name'].split(':')[1]
        plot_pair_residual(data_dir, code1, code2, output_dir, start_date='2020-01-01')


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
        search_pair_trade(data_dir, automotive_codes, automotive_names, output_dir, p_threshold=0.01)
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
    plot_pair_residual(args['data_dir'], args['pred_stock_code1'], args['pred_stock_code2'], args['output_dir'], start_date=pred_start)


if __name__ == '__main__':
    matplotlib.use('Agg')  # GUI使えない場合はこれつける
    plot_pair_trade()  # 2銘柄でグラフ可視化

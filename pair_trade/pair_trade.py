# -*- coding: utf-8 -*-
"""
ペアトレードに向いている銘柄を探索
※ペアトレード：安な銘柄を「買い」、割高な銘柄を「空売り」しておき、株価が近づいたタイミングで反対売買を行い利益を確定する

<具体的な処理>
1. 1銘柄の収益率の累積積について、ADF検定を実施し、p値が小さい銘柄同士でクラスタリング
2. p値<0.1のペアの残差について、線形回帰モデル作成する
- p値の距離が近い銘柄同士は共和分(=非定常のデータの足し算は定常過程になる性質)の関係にあると解釈
- 共和分により、その2銘柄の収益率の累積積の差（残差）は0を基準に行ったり来たりするはずなので。ペアトレードに向いている銘柄といえる

※ADF検定は単位根過程(=非定常だけど、差分をとると（弱）定常になるようなデータ)かどうかの検定
　p値<0.1なら単位根過程でないと解釈する

2銘柄の累積収益率の残差グラフに周期性があるか確認する必要があるためnotebookで実行すべき
参考:run_pair_trade.ipynb

予測のみ実行する場合は以下のコマンドで実行できる
    $ activate stock
    $ python pair_trade.py -m output/lr_residual_7201_7202.joblib -c1 7201 -c2 7202  # 7201,7202について累積収益率の残差グラフ作成
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # グラフの体裁きれいにする
plt.rcParams["font.family"] = 'Yu Gothic'   # Yu Gothic指定すれば日本語出せる


class Cluster():

    def plot_dendrogram(self, df, method='ward', figsize=(15, 8), output_dir=None):
        """
        階層型クラスタリングで樹形図（デンドログラム）と距離行列のヒートマップをplotする
        Usage:
            import seaborn as sns
            df = sns.load_dataset('iris')
            df = df.drop("species", axis=1)  # 数値列だけにしないとエラー
            df_dist, df_z = plot_dendrogram(df.T, method='ward')  # データフレーム転置しないと列についてのクラスタリングにはならない
        """
        import seaborn as sns
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

        # 数値列だけにしないと距離測れない
        _df = df.T
        cols = [col for col in _df.columns if _df[col].dtype.name in ['object', 'category', 'bool']]
        assert len(cols) == 0, '数値以外型の列があるので階層型クラスタリングできません'

        # 階層型クラスタリング実行
        # クラスタリング手法である methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]　ある
        z = linkage(df, method=method, metric='euclidean')
        # linkage関数は、クラスタリングのステップ毎に
        # [クラスター番号1, クラスター番号2, クラスタリングに伴う距離の変化, 新しいクラスターに属するデータ数] を返す
        df_z = pd.DataFrame(z, columns=['index1', 'index2', 'distance', 'date_count'])
        df_z['index1_name'] = df_z['index1'].apply(lambda x: df.index[int(x)] if int(x) < len(df) else np.nan)
        df_z['index2_name'] = df_z['index2'].apply(lambda x: df.index[int(x)] if int(x) < len(df) else np.nan)

        # デンドログラムを描く
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(z, labels=df.index, ax=ax)
        plt.title(method)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, method + '_dendro.png'), bbox_inches="tight")
        plt.show()

        # 距離行列計算
        s = pdist(df)
        df_dist = pd.DataFrame(squareform(s), index=df.index, columns=df.index)  # 距離行列を平方行列の形にする
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df_dist, cmap='coolwarm', annot=True, ax=ax)
        plt.title('distance matrix')
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, method + '_distmatrix.png'), bbox_inches="tight")
        plt.show()

        # クラスタリング手法の評価指標計算 値大きい方が高精度らしい https://www.haya-programming.com/entry/2019/02/11/035943
        c, d = cophenet(z, s)
        print('method:{0} {1:.3f}'.format(method, c))

        return df_dist, df_z

    def calc_adf_test_return_rate_cumprod(self, csv_path: str, train_start_date=None):
        """ 1銘柄の株価csvから収益率の累積積を計算してADF検定のp値を返す """
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
        df_return_rate = df_return_rate.apply(np.cumprod).dropna()

        # ADF検定のp値返す
        ts = df_return_rate.iloc[:, 0]
        dftest = sm.tsa.stattools.adfuller(ts, autolag='AIC')
        return dftest[1], df_return_rate

    def cluster_adf_p_val(self, data_dir: str, codes: list, names: list, train_start_date=None, figsize=(15, 8), output_dir=None):
        """ calc_adf_test_return_rate_cumprod(), plot_dendrogram() 一気に実行 """
        # 指定銘柄でADF検定のp値計算
        adf_p_vals = []
        merge_df_return_rate = None
        for code in codes:
            csv_path = os.path.join(data_dir, f'{code}.csv')
            # calc_adf_test_return_rate_cumprod()
            adf_p_val, df_return_rate = self.calc_adf_test_return_rate_cumprod(csv_path, train_start_date=train_start_date)
            adf_p_vals.append(adf_p_val)
            merge_df_return_rate = df_return_rate if merge_df_return_rate is None else merge_df_return_rate.join(df_return_rate, how='outer')

        df_adf_p_vals = pd.DataFrame({'name': names, 'code': codes, 'adf_p_val': adf_p_vals})
        df_adf_p_vals['name_code'] = df_adf_p_vals['name'] + ':' + df_adf_p_vals['code'].astype(str)
        #print(df_adf_p_vals)

        # ADF検定のp値でクラスタリング
        _df = df_adf_p_vals[['name_code', 'adf_p_val']]
        _df = _df.set_index('name_code')
        # plot_dendrogram()
        df_dist, df_z = self.plot_dendrogram(_df, method='single', figsize=figsize, output_dir=output_dir)
        #print(df_dist)

        return merge_df_return_rate, df_dist, df_z


class Model():

    def make_residual_lr_model(self, merge_df_return_rate, code1: str, code2: str, figsize=(15, 8), output_dir=None):
        """ 2銘柄の累積収益率の残差について、線形回帰モデル作成 """
        # 2銘柄の累積収益率の残差
        _df_return_rate = merge_df_return_rate[[code1, code2]].dropna()
        _df_return_rate['resid'] = _df_return_rate[code1] - _df_return_rate[code2]
        X = _df_return_rate[[code1, code2]]  # 特徴量(2銘柄の累積収益率)
        Y = _df_return_rate['resid']  # 目的変数（2銘柄の累積収益率の残差）

        # 一応トレーニング・テストデータ分割
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # LinearRegression  y = (1)*x1 + (-1)*x2
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # 予測　
        Y_pred = model.predict(X_test)

        # 平均絶対誤差(MAE) 正解値と予測値の差分の絶対値の平均
        mae = mean_absolute_error(Y_test, Y_pred)
        # 平方根平均二乗誤差（RMSE）正解値と予測値の差分の二乗を平均し、平方したもの
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        # スコア
        score = model.score(X_test, Y_test)

        print("MAE = %.2f,  RMSE = %.2f,  score = %.2f" % (mae, rmse, score))
        print("Coef:重み = ", model.coef_)
        print("Intercept:バイアス項 =", model.intercept_)

        df_pred = pd.DataFrame({'Y_pred':Y_pred, 'date':Y_test.index})
        df_pred['date'] = pd.to_datetime(df_pred['date'], format='%Y-%m-%d')

        fig, ax = plt.subplots(figsize=figsize)  # subplotでfigsize指定
        Y.plot(ax=ax)
        df_pred.plot(x='date', y='Y_pred', ax=ax)
        plt.legend()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'lr_residual.png'), bbox_inches="tight")
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
        print("INFO: save file. {}".format(filepath))
        return save_data, filepath

    def predict_load_model(self, model_path: str, X_test, out_png=None):
        """ 作成したモデルファイルロードしてpredict """
        model_info = joblib.load(model_path)

        Y_pred = model_info['trained_model'].predict(X_test)
        df_pred = pd.DataFrame({'Y_pred': Y_pred, 'date': X_test.index})
        df_pred['date'] = pd.to_datetime(df_pred['date'], format='%Y-%m-%d')

        # 予測のグラフ表示
        fig, ax = plt.subplots(figsize=(15, 8))  # subplotでfigsize指定
        df_pred.plot(x='date', y='Y_pred', ax=ax)
        plt.title('2銘柄の累積収益率の残差')
        if out_png is not None:
            plt.savefig(out_png, bbox_inches="tight")
        plt.show()

        return df_pred

    def run_pair_make_residual_lr_model(self, merge_df_return_rate, c1: str, c2: str, output_dir: str, pred_start_date: str):
        """ 銘柄のペアを変えてモデル作成 """
        model = self.make_residual_lr_model(merge_df_return_rate, c1, c2, output_dir=output_dir)

        # モデル保存
        training_info = {'training_data': f'{c1}.csv_{c2}.csv', 'model_type': 'LinearRegression', 'hyper_pram': 'default'}
        save_data, filepath = self.save_model_info(model,
                                                   training_info,
                                                   preprocess_pipeline=None,
                                                   output_path=os.path.join(output_dir, f'lr_residual_{c1}_{c2}.joblib'))

        # モデルロードして予測
        _df_return_rate = merge_df_return_rate[[c1, c2]].dropna()
        _ = self.predict_load_model(filepath,
                                    _df_return_rate[pred_start_date:],  # _df_return_rate['2020-01-01':],
                                    out_png=os.path.join(output_dir, f'lr_residual_{c1}_{c2}.png'))


def search_pair_trade(data_dir, codes, names, output_dir, p_threshold=0.01, train_start_date='2010-01-01'):
    """ ペアトレードに向いている銘柄を探索 """
    # calc_adf_test_return_rate_cumprod(), plot_dendrogram() 一気に実行
    merge_df_return_rate, df_dist, df_z = Cluster().cluster_adf_p_val(data_dir, codes, names,
                                                                      train_start_date=train_start_date,
                                                                      output_dir=output_dir)

    # ペアのクラスター + p値が小さいペア確認
    df_pair = df_z[(df_z['date_count'] == 2.0) & (df_z['distance'] < p_threshold)]

    # 残差の線形モデル作る  y = (1)*x1 + (-1)*x2
    for ind, row in df_pair.iterrows():
        code1 = row['index1_name'].split(':')[1]
        code2 = row['index2_name'].split(':')[1]
        print(f"---- {row['index1_name']}, {row['index2_name']} -----")
        Model().run_pair_make_residual_lr_model(merge_df_return_rate, code1, code2, output_dir, '2020-01-01')


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
        # データロード
        output_dir = r'D:\work\02_keras_py\experiment\01_code_test\output_test\tmp'
        data_dir = r'D:\DB_Browser_for_SQLite\csvs\kabuoji3'
        automotive_names = ['イーソル', '日産自', 'いすゞ', 'トヨタ', '日野自', '三菱自', 'マツダ', 'ホンダ', 'スズキ', 'ＳＵＢＡＲＵ']
        automotive_codes = [4420, 7201, 7202, 7203, 7205, 7211, 7261, 7267, 7269, 7270]

        # ペアトレードに向いている銘柄を探索
        search_pair_trade(data_dir, automotive_codes, automotive_names, output_dir)
    _test_search_pair_trade()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--pred_model_path", type=str, default=None, help="model info paths")
    ap.add_argument("-o", "--output_dir", default=r"output", help="output dir path")
    ap.add_argument("-d", "--data_dir", default=r"D:\DB_Browser_for_SQLite\csvs\kabuoji3", help="stock csv dir path")
    ap.add_argument("-c1", "--pred_stock_code1", type=int, default=7201, help="predict stock code1")
    ap.add_argument("-c2", "--pred_stock_code2", type=int, default=7202, help="predict stock code2")
    ap.add_argument("-p_s", "--pred_start_date", type=str, default=None, help="predict start date")
    return vars(ap.parse_args())


def pred_pair_trade():
    """ 作成したモデルで予測のみ実行する """
    args = get_args()
    os.makedirs(args['output_dir'], exist_ok=True)

    # 予測結果の表示開始日
    now = datetime.datetime.now()  # date型にする場合は datetime.date.today()
    pred_start = now - datetime.timedelta(days=20) if args['pred_start_date'] is None else args['pred_start_date']
    pred_start = pred_start.strftime('%Y-%m-%d')  # => '2019-08-02'

    # データ前処理
    codes = [args['pred_stock_code1'], args['pred_stock_code2']]
    names = [str(args['pred_stock_code1']), str(args['pred_stock_code2'])]
    merge_df_return_rate, df_dist, df_z = Cluster().cluster_adf_p_val(args['data_dir'], codes, names, output_dir=None)
    X = merge_df_return_rate.dropna()

    # 予測のみ実行
    _ = Model().predict_load_model(args['pred_model_path'],
                                   X[pred_start:],
                                   out_png=os.path.join(args['output_dir'],
                                                        f"lr_residual_{str(args['pred_stock_code1'])}_{str(args['pred_stock_code2'])}.png"))


if __name__ == '__main__':
    matplotlib.use('Agg')  # GUI使えない場合はこれつける
    pred_pair_trade()  # 予測のみ実行

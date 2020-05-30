# -*- coding: utf-8 -*-
"""
時系列データ(ts:time_series)に対して、
SARIMAモデル(Seasonal ARIMA model：周期的な季節変動の効果を追加したARIMAモデル)作成

Usage:
    $ activate tfgpu20
    $ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30  # train/test setを120行目で分け、パラメータチューニング30回実行し、モデル作成
    $ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30 -fix_s 12  # 周期性sを固定してパラメータチューニングしてモデル作成
    $ python sarimax_analysis.py -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01  # 予測のみ
    $ python sarimax_analysis.py -i AirPassengers.csv -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01  # 予測とtrain dataもplot
    $ python sarimax_analysis.py -i AirPassengers.csv -p_m output/SARIMAX_best.joblib -p_s 145 -p_e 160  # 予測とtrain dataもplot。うまくいかず。。。

動作確認につかったnotebook: test_sarimax_analysis.py.ipynb
データによって周期性や階差分違うから、モデル作成についてはnotebookで自己相関plotを確認しながら実行した方がホントはよさそう
"""
import argparse
import os
import joblib
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # グラフの体裁きれいにする

import optuna


def plot_acf_pacf(ts: pd.Series, lags=48, out_png=None):
    """
    自己相関（ACF）と偏自己相関（PACF）のグラフを縦に並べてplot
    - 自己相関:時刻tとt-nとの相関 = 過去の値が現在のデータにどれくらい影響しているか
    - 偏自己相関:ある時点同士だけの関係性。
      今日と二日前の関係には間接的に一日前の影響が含まれるが、一日前の影響を除いて今日と二日前だけの関係を調べられる
    Usage:
        plot_acf_pacf(ts)
    """
    if len(ts) < lags:
        lags = len(ts) - 1
    # 自己相関(acf)のグラフ
    # 薄い赤で塗り潰された領域は95%信頼区間。
    # 信頼区間の領域を超えてプロットされているデータは、統計的に有意差がある値とみれる
    # 信頼区間におさまっていれば自己相関を持たない、つまり現在の自分と過去の自分は関係ないということ
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=lags, ax=ax1)

    # 偏自己相関(pacf)のグラフ
    # 信頼区間におさまっていれば自己相関を持たない、つまり現在の自分と過去の自分は関係ないということ
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2)
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
    plt.show()


def plot_seasonal_decompose(ts: pd.Series, c='r', out_png=None, freq=None):
    """
    オリジナル ->トレンド成分、季節成分、残差成分に分解してプロット
    - 季節性(Seasonality)：周期的に繰り返す変動。四季とは直接関係なくても周期的な変動は季節性と表現する
    - トレンド(Trend)：右方上がりに増えているみたいなデータの傾向のこと。実データから季節性を引き算したデータに対応する。要は移動平均的な線のこと
    - 残差(Residual)：トレンドと季節性を除いたその他変動成分
    Usage:
        res = plot_seasonal_decompose(ts)
    """
    res = sm.tsa.seasonal_decompose(ts, freq=freq)

    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    plt.figure(figsize=(8, 8))

    # オリジナルの時系列データプロット
    # res.plot() で3グラフ一気にplotできるが、グラフサイズ調整するためsubplotで各グラフ描いてる
    plt.subplot(411)
    plt.plot(ts, c=c)
    plt.ylabel('Original')

    # trend のプロット
    plt.subplot(412)
    plt.plot(trend, c=c)
    plt.ylabel('Trend')

    # seasonal のプロット
    plt.subplot(413)
    plt.plot(seasonal, c=c)
    plt.ylabel('Seasonality')

    # residual のプロット
    plt.subplot(414)
    plt.plot(residual, c=c)
    plt.ylabel('Residuals')

    plt.tight_layout()  # グラフ間スキマ調整

    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
    plt.show()

    return res


def plot_stationarity(ts: pd.Series, window_size=12, output_dir=None):
    """
    定常性を確認する Augmented Dickey-Fuller test(ADF検定) 結果と標準偏差、平均のプロット
    https://qiita.com/mshinoda88/items/749131478bfefc9bf365
    Usage:
        test_stationarity(ts, window_size=12)
    """
    # Determing rolling statistics
    rolmean = ts.rolling(window=window_size, center=False).mean()
    rolstd = ts.rolling(window=window_size, center=False).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(10, 8))
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'stationarity.png'), bbox_inches="tight")
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = sm.tsa.stattools.adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value',
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if output_dir is not None:
        pd.DataFrame(dfoutput).to_csv(os.path.join(output_dir, 'stationarity.csv'))


# 損失関数をAICではなく残差とする場合につかう
def make_lbtest_df(resid, lags=10):
    """
    残差に対するLjung-Box検定
    https://blog.brains-tech.co.jp/entry/arima-tutorial-2
    良いモデルかどうかは、作成したモデルから出力される結果と実際の値との残差が、
    平均0、自己相関係数が全て0のホワイトノイズになっているかどうかで判断
    Usage:
        print(model.resid.mean())  # train setの残差平均確認
        make_lbtest_df(model.resid)
    """
    if len(resid) < lags:
        lags = len(resid) - 1
    test_result = sm.stats.diagnostic.acorr_ljungbox(resid, lags=lags)
    lbtest_df = pd.DataFrame({"Q": test_result[0],"p-value": test_result[1]})
    lbtest_df = lbtest_df[["Q", "p-value"]]
    return lbtest_df


class SarimaxObjective(object):
    """
    SARIMAモデルのパラメータチューニング
    Usage:
        run_optuna()参照
    """
    def __init__(self, ts, fix_s):
        self.ts = ts
        self.fix_s = fix_s

    def get_sarimax_parameter_suggestions(self, trial) -> dict:
        """
        SARIMAモデルのパラメータ
        全パラメータ探索できるようにしているが、自己相関のグラフを確認し、周期性打ち消せる階差分dや季節周期sを把握して、d,sある程度固定して実行した方がいい
        特に季節周期sはランダムだとエラーになりまくる
        """
        # 季節周期sはランダムだとエラーになりまくるので固定値にできるようにしておく
        s_suggest_int = trial.suggest_int('s', 0, 30) if self.fix_s is None else trial.suggest_int('s', self.fix_s, self.fix_s)
        return {
            'trend': trial.suggest_categorical('trend', ['n', 'c', 't', 'ct', None]),
            # p時点過去までの値を使うARモデルをAR(p)
            'p': trial.suggest_int('p', 1, 4),  # AR項が0だとエラーだから1スタート
            # ARIMAのd階差分
            'd': trial.suggest_int('d', 0, 4),
            # q時点過去までのノイズを使うMAモデルをMA(q)
            'q': trial.suggest_int('q', 0, 4),
            # SARIMAの季節周期s
            's': s_suggest_int,  # AirPassengers.csvの場合は12ヵ月周期
            # 季節差分方向のARIMA(P,D,Q)のP
            'P': trial.suggest_int('P', 0, 2),
            # 季節差分方向のARIMA(P,D,Q)のD
            'D': trial.suggest_int('D', 0, 2),
            # 季節差分方向のARIMA(P,D,Q)のQ
            'Q': trial.suggest_int('Q', 0, 2),
            # 推定されるAR部分が定常性を持つように強制するか否か  Trueだとエラー乱発するのでFalseにしておく
            'enforce_stationarity': trial.suggest_categorical('enforce_stationarity', [False, False]),  # 'enforce_stationarity': trial.suggest_categorical('enforce_stationarity', [True, False]),
            # 推定されるMA部分が反転可能性を持つように強制するか否か  Trueだとエラー乱発するのでFalseにしておく
            'enforce_invertibility': trial.suggest_categorical('enforce_invertibility', [False, False])  # 'enforce_invertibility': trial.suggest_categorical('enforce_invertibility', [True, False])
        }

    def __call__(self, trial):
        try:
            params = self.get_sarimax_parameter_suggestions(trial)
            model = sm.tsa.SARIMAX(self.ts,
                                   trend=params['trend'],
                                   order=(params['p'], params['d'], params['q']),  # order=(p, d, q)。pはAR（自己回帰）、qはMA（移動平均）、dが差分の次数
                                   seasonal_order=(params['P'], params['D'], params['Q'], params['s']),  # seasonal_order=(P, D, Q, s)。 順番はP,D,Q,sで前半の3つorderと一緒、最後4つ目は周期。季節調整に適用する各々の次数を指定
                                   enforce_stationarity=params['enforce_stationarity'],  # 定常性強制
                                   enforce_invertibility=params['enforce_invertibility']  # 可逆性強制
                                  ).fit()
            return model.aic  # AICで最適化
        except Exception as e:
            traceback.print_exc()  # Exceptionが発生した際に表示される全スタックトレース表示
            return e


def run_optuna(ts_train: pd.Series, output_dir: str, n_trials=500, fix_s=None):
    """optunaでパラメータチューニング実行"""
    study = optuna.create_study(direction='minimize')  # 最小化
    study.optimize(SarimaxObjective(ts_train, fix_s), n_trials=n_trials)

    print(f"試行回数: {len(study.trials)}")
    print(f"目的関数を最小化するbestパラメータ: {study.best_params}")
    print(f"目的関数の最小値: {study.best_value}")
    df_result = study.trials_dataframe().sort_values(by='value')
    df_result.to_csv(os.path.join(output_dir, 'result_optuna.csv'), index=False)
    return study


def save_model_info(model, training_info: dict, preprocess_pipeline=None, output_path='.'):
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


def predict_load_model(model_path: str, strat_date, end_date, ts=None, out_png=None):
    """
    作成したモデルファイルロードしてpredict。95%信頼区間のグラフplotも出す
    strat_date, end_dateは'2020-05-12'みたいなdate形式以外に、123のような番号でも可能
    """
    model_info = joblib.load(model_path)

    # 予測の数値だけでよいなら ts_pred = model_info['trained_model'].predict('1958-01-01', '1960-12-01') でいける
    ts_pred_obj = model_info['trained_model'].get_prediction(strat_date, end_date)
    ts_pred_ci = ts_pred_obj.conf_int(alpha=0.05)  # 信頼区間取得 defalut alpah=0.05 :returns a 95% confidence interval

    # グラフ表示
    plt.figure(figsize=(12, 4))
    if ts is not None:
        plt.plot(ts, label="actual")  # 実データプロット
    plt.plot(ts_pred_obj.predicted_mean, c="b", linestyle='--', label="model-pred", alpha=0.7)  # 予測プロット

    # 予測の95%信頼区間プロット（帯状）
    plt.fill_between(ts_pred_ci.index, ts_pred_ci.iloc[:, 0], ts_pred_ci.iloc[:, 1], color='g', alpha=0.2)
    plt.legend(loc='upper left')

    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")

    plt.show()

    return ts_pred_obj.predicted_mean


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_csv", default=None, help="paths to input time_series csv")
    ap.add_argument("-o", "--output_dir", default=r"output", help="output dir path")
    ap.add_argument("-s", "--split_n_train_test", type=int, default=120, help="split train/test set count")  # AirPassengers.csvに合わせてtrain set=120個,test setそれ以外として分けた
    ap.add_argument("-n_t", "--n_trials", type=int, default=500, help="optuna n_trials")
    ap.add_argument("-fix_s", "--fix_param_s", type=int, default=None, help="fix param s")
    ap.add_argument("-p_m", "--pred_model_path", default=None, help="predict model path")
    ap.add_argument("-p_s", "--pred_start", default=r"1958-01-01", help="predict start date")
    ap.add_argument("-p_e", "--pred_end", default=r"1960-12-01", help="predict end date")
    return vars(ap.parse_args())


if __name__ == '__main__':
    matplotlib.use('Agg')  # GUI使えない場合はこれつける

    args = get_args()
    os.makedirs(args['output_dir'], exist_ok=True)

    if args['input_csv'] is not None:
        df = pd.read_csv(args['input_csv'], index_col=0, parse_dates=[0], dtype='float')  # 0列目をdatetime型でロードしてindexにする
        ts = df.iloc[:, 0]  # 1列目を時系列データとする。0を指定しているのは0列目はindexにしたから
    else:
        ts = None

    if args['pred_model_path'] is not None:
        # 予測のみ
        ts_pred = predict_load_model(args['pred_model_path'],
                                     args['pred_start'],
                                     args['pred_end'],
                                     out_png=os.path.join(args['output_dir'], 'ts_pred.png'),
                                     ts=ts
                                     )
        pd.DataFrame(ts_pred).to_csv(os.path.join(args['output_dir'], 'ts_pred.csv'))

    else:
        # オリジナル ->トレンド成分、季節成分、残差成分に分解してプロット
        res = plot_seasonal_decompose(ts, out_png=os.path.join(args['output_dir'], 'seasonal_decompose.png'))

        # 定常性確認=ADF検定
        # 12ヵ月周期として、window_size=12とする
        plot_stationarity(ts, window_size=12, output_dir=args['output_dir'])

        # 自己相関確認
        plot_acf_pacf(ts, out_png=os.path.join(args['output_dir'], 'acf_pacf.png'))

        # train/test set にデータ分割
        # 周期性に合わせて分けないとうまくいかないことがあった
        ts_train = ts[:args['split_n_train_test']]
        ts_test = ts[args['split_n_train_test']:]

        # SARIMAXパラメータチューニング
        study = run_optuna(ts_train, args['output_dir'], n_trials=args['n_trials'], fix_s=args['fix_param_s'])

        # bestパラメでモデル作成
        model = sm.tsa.statespace.SARIMAX(ts_train,
                                          trend=study.best_params['trend'],
                                          order=(study.best_params['p'], study.best_params['d'], study.best_params['q']),
                                          seasonal_order=(study.best_params['P'], study.best_params['D'], study.best_params['Q'], study.best_params['s']),
                                          enforce_stationarity=study.best_params['enforce_stationarity'],
                                          enforce_invertibility=study.best_params['enforce_invertibility']
                                          ).fit()
        print('model.summary(): ', model.summary())

        # モデル保存
        training_info = {'training_data': args['input_csv'], 'model_type': 'SARIMAX', 'hyper_pram': study.best_params}
        save_data, filepath = save_model_info(model,
                                              training_info,
                                              preprocess_pipeline=None,
                                              output_path=os.path.join(args['output_dir'], 'SARIMAX_best.joblib'))
        load_data = joblib.load(filepath)

        # 予測
        ts_pred = predict_load_model(filepath,
                                     ts_test.index[0].strftime('%Y-%m-%d'),
                                     ts_test.index[-1].strftime('%Y-%m-%d'),
                                     ts=ts,
                                     out_png=os.path.join(args['output_dir'], 'ts_pred.png'))

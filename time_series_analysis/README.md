# time_series_analysis
時系列データ(ts:time_series)に対して、AR/ARMA/ARIMA/SARIMAモデル(Seasonal ARIMA model：周期的な季節変動の効果を追加したARIMAモデル)作成
- データによって周期性や階差分違うから、モデル作成についてはnotebookで自己相関plotを確認しながら実行した方がよさそう
- 参考: test_sarimax_model.py.ipynb, test_AR_ARMA_ARIMA_SARIMA.ipynb

## Usage
```bash
$ activate tfgpu20
$ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30                                                   # train/test setを120行目で分け、パラメータチューニング30回実行し、SARIMAモデル作成
$ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30 -fix_s 12                                         # 周期性sを固定してパラメータチューニングしてSARIMAモデル作成
$ python sarimax_analysis.py -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01                       # 予測のみ
$ python sarimax_analysis.py -i AirPassengers.csv -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01  # 予測とtrain dataもplot
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)
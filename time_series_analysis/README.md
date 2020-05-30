# time_series_analysis
���n��f�[�^(ts:time_series)�ɑ΂��āAAR/ARMA/ARIMA/SARIMA���f��(Seasonal ARIMA model�F�����I�ȋG�ߕϓ��̌��ʂ�ǉ�����ARIMA���f��)�쐬
- �f�[�^�ɂ���Ď�������K�����Ⴄ����A���f���쐬�ɂ��Ă�notebook�Ŏ��ȑ���plot���m�F���Ȃ�����s���������悳����
- �Q�l: test_sarimax_model.py.ipynb, test_AR_ARMA_ARIMA_SARIMA.ipynb

## Usage
```bash
$ activate tfgpu20
$ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30                                                   # train/test set��120�s�ڂŕ����A�p�����[�^�`���[�j���O30����s���ASARIMA���f���쐬
$ python sarimax_analysis.py -i AirPassengers.csv -s 120 -n_t 30 -fix_s 12                                         # ������s���Œ肵�ăp�����[�^�`���[�j���O����SARIMA���f���쐬
$ python sarimax_analysis.py -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01                       # �\���̂�
$ python sarimax_analysis.py -i AirPassengers.csv -p_m output/SARIMAX_best.joblib -p_s 1959-01-01 -p_e 1960-12-01  # �\����train data��plot
```

<!-- 
## License
This software is released under the MIT License, see LICENSE.
-->

## Author
- Github: [riron1206](https://github.com/riron1206)
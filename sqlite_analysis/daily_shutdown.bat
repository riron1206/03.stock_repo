@rem �쐬��2020/04/23 DB�����X�Vbat���s���āA�I��������PC���Ƃ�

@rem daily.bat�Ăяo��
cd C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\sqlite_analysis\create_db
call daily.bat

@rem order.csv�쐬
cd C:\Users\81908\jupyter_notebook\stock_work\03.stock_repo\sqlite_analysis\auto_trade
call 01.run_make_order_csv.bat

@rem �V���b�g�_�E���R�}���h ���b�Z�[�W���o�͂�60�b��Ƀp�\�R������܂�
shutdown /s /t 60 /c "��60�b�ŃV���b�g�_�E�����܂��B"


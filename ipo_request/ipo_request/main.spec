# -*- mode: python ; coding: utf-8 -*-

### 個人の./libsを認識するためのコード ###
# https://qiita.com/cheuora/items/39b3203400e1e15248ed
import sys 
myLibPath = './libs'
sys.path.append(myLibPath)
##########################################

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\shingo\\jupyter_notebook\\stock_work\\03.stock_repo\\ipo_request\\ipo_request'],
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )

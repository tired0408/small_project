# coding=gbk
import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {'packages': [], 'excludes': [],"packages"  : ["os"]}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

iconpath = "Ico\logo.ico"

setup(  name = 'main',
        version = '0.01',
        description = '作者'.decode('gb2312'),
        options = {'build_exe': build_exe_options},
        executables = [Executable('main.py', base=base, icon=iconpath)])



#这个是cx_Freeze的生成工具
#CMD 下进入该目录，然后运行 python setup.py build

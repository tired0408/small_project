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
        description = '����'.decode('gb2312'),
        options = {'build_exe': build_exe_options},
        executables = [Executable('main.py', base=base, icon=iconpath)])



#�����cx_Freeze�����ɹ���
#CMD �½����Ŀ¼��Ȼ������ python setup.py build

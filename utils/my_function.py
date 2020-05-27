import os
import psutil
import sys
import time
import shutil
def restart_program():
    """
    重启该进程
    防止：程序出现错误时候，或误点击Ctrl+C导致的错误退出
    :return:
    """
    def restart():
        try:
            p = psutil.Process(os.getpid())
            print(os.getpid())
            for handler in p.open_files() + p.connections():
                os.close(handler.fd)
        except Exception as e:
            print(e)
        python = sys.executable
        os.execl(python, python, *sys.argv)
    i = 1
    try:
        while 1:
            i +=1
            time.sleep(1)
            print("yyyyyyyyyyyyyyyyy", i)
            if i >10:
                raise Exception("xxxxxxxxxxxxxxxxxxxx")
    except:
        print('restart')
        restart()

def the_iterator():
    """
    迭代器
    :return:
    """
    def readInChunks(fileObj, chunkSize=1024*1024*100):
        while 1:
            data = fileObj.readlines(chunkSize)
            if not data:
                break
            yield "".join(data)
    def cut_file(path):
        f = open(path)
        export_path = "./data/totalExposureLog/"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        os.mkdir(export_path)
        for chuck in readInChunks(f):
            wrfile=open(export_path+"totalExposureLog_"+str(i)+".out",'w')
            wrfile.write(chuck)
            wrfile.close()
        f.close()

import cx_Oracle
def use_oracle():
    """
    操作oracle数据库的相关方法
    :return:
    """
    # 连接数据库，path格式[用户名]/[密码]@[host]:[port]/[SERVICE_NAME]
    path = "ideal/ideal@192.101.109.110:1521/orcl"
    db = cx_Oracle.connect(path)
    # 获取指针
    cursor = db.cursor()
    # 执行语句
    cursor.execute("""
            select * from hujt_test 
            """)
    # 获取全部数据
    all_data = cursor.fetchall()
    # 逐条获取数据
    one_data = cursor.fetchone()
    db.close()

def singleton(cls):
    """
    装饰器实现单例模式
    :param cls: 实例：方法或者类
    :return:
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance
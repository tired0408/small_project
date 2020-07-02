from utils.my_function import *
from utils.my_method import *
from utils.problem import *
####################################
######## 常用基础使用方法
####################################
normal()  # 基础方法：时间处理、字节处理、排序、指定GPU使用
use_matlab()  # 使用matplotlib画图
pandas_method() # pandas的使用方法，操作CSV,EXCEL等
numpy_method() # numpy的使用方法
regular_expression()  # 正则表达式
opencv_note()  # opencv的学习
####################################
######## 特别的一些功能
####################################
restart_program()  # 重启程序
the_iterator()  # 迭代器：逐步读取文件，避免内存占用过大
use_oracle()  # Oracle数据库操作方法
singleton(object) # 装饰器实现单例模式：一个程序只需要实现一次

####################################
######## 容易出现的BUG点
####################################
problem1() # 默认参数值在函数定义时只计算一次，这意味着修改参数的默认值将影响函数的所有后续调用。
problem2() # 删除元素降低内存
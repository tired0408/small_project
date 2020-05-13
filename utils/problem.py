import numpy as np
# 默认参数值在函数定义时只计算一次，这意味着修改参数的默认值将影响函数的所有后续调用。
def cache(lt=np.zeros((3, 3))):
    print(lt)
    lt[1,1] = 1
cache()
cache()
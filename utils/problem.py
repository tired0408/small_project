import numpy as np
import sys
import time
import os
import psutil
import cv2
# 默认参数值在函数定义时只计算一次，这意味着修改参数的默认值将影响函数的所有后续调用。
def problem1():
    def cache(lt=np.zeros((3, 3))):
        print(lt)
        lt[1,1] = 1
    cache()
    cache()
# 消减元素并不会释放内存
def problem2():
    # 全局变量内存
    bgr_image = np.zeros((1920, 1080, 3), dtype=np.uint8)  # linux减少,windows减少
    # bgr_image = cv2.imread("test.jpg")  # linux一段时间后减少,windows减少
    history_video = [bgr_image.copy() for i in range(250)]
    while 1:
        time.sleep(0.04)
        if len(history_video) > 0:
            history_video.pop(0)
        else:
            break
    # 类中的变量内存
    class A:
        def __init__(self):
            print("init")
            # windows减少，linux不减少
            # linux不减少可能为，linux判断后续还要用到内存，将该释放的内存转移过去
            self.bgr_image = cv2.imread("test.jpg")
            # self.bgr_image = np.zeros((1920,1080,3), dtype=np.uint8) # windows、linux减少,
            self.history_video = [self.bgr_image.copy() for i in range(250)]

        def del_list(self):
            print("del list start")
            while 1:
                time.sleep(0.04)
                if len(self.history_video) > 0:
                    self.history_video.pop(0)
                else:
                    break
            print("finnish del list")
    a = A()
    a.del_list()
    time.sleep(20)

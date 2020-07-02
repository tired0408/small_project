#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import numpy as np

'''
    * 投放时段：字符串。包含 7 个 64 位无符号整型数字（逗号分隔），
		每个整数分别代表周一到周日的投放时段。该整数转为 2 进制后从低到高 48 位bit 
		代表全天各时段（半小时为一时间窗口）是否投放，1-投放，0-不投。
	
		举例说明:  17179865088 = 1111111111111111111111000000000000，代表投放时段为 6:00-17:00；
		
		11 11 11 11 11 11 11 11 11 11 11 00 00 00 00 00 00
		16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0

		281474976710655=111111111111111111111111111111111111111111111111，代表全天投放。


'''


#把数字转换成二进制，并拆分出每一位，逆序后形成一个List
def long2bit (n):
    pass
    #转成二进制,前面再补0到48位
    t = bin(n)[2:].zfill(48)
    lstRet = list(t)
    lstRet.reverse()
    return lstRet

#把投放时段转换成List,二层; 
#投放时段：字符串。包含 7 个 64 位无符号整型数字（逗号分隔）
#测试：
# t =  "17179865088,17179865088,4294950912,17109865088,17109865088,281474976710655,281474976710655"
# t = '17179865088,281474976710655'
#print(Period2List (t))  
#[['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']]
def Period2List (strPeriod):
    pass
    lstT = strPeriod.split(',')
    lstR = []
    for x in lstT:
        lstR.append(long2bit(int(x)))
    return lstR

t = '17179865088,281474976710655'
print(Period2List (t))

if __name__ == '__main__':
    pass


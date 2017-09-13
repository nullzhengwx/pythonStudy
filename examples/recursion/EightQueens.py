# -*- coding:utf-8 -*-
__metaclass__ = type
import random

"""
yield的作用:
函数会被冻结: 即函数停在那点等待被重新唤醒.函数被唤醒后就从停止的那点开始执行.
"""

def conflict(state, nextX):
    """
    判断某一行的皇后的位置是否和前面的皇后冲突
    :param state:   前面的皇后的位置, 原组 state[0] = 3表示第一行,第四列
    :param nextX:   某一行皇后的位置
    :return:
    """
    nextY = len(state)
    for i in range(nextY) :
        if abs(state[i] - nextX) in (0, nextY - i) :
            return True
    return False

def queens(num=8, state=()):
    """
    recursion!
    从前面的皇后得到了包含位置信息的元组,并且要为后面的皇后提供当前皇后的每种合法的位置信息
    :param num:
    :param state:
    :return:
    """
    for pos in range(num):
        if not conflict(state, pos) :
            if(len(state) == num - 1):      # 基本的情况
                yield (pos,)
            else :                          # 递归
                for result in queens(num, state + (pos,)) :
                    yield (pos,) + result

def prettyPrint(solution):
    def line(pos, length=len(solution)) :
        return ". " * (pos) + "X " + ". " * (length - pos - 1)

    for pos in solution:
        print line(pos)

print list(queens(3))
print list(queens(4))
print list(queens(5))

prettyPrint(random.choice(list(queens(8))))
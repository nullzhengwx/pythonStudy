import timeit

"""according to
https://github.com/fluentpython/example-code/blob/master/02-array-seq/listcomp_speed.py"""
TIMES = 10000

SETUP = """
symbols = '$¢£¥€¤'
def non_ascii(c):
    return c > 127
"""

""" training and testing"""
x = 'ABC'
dumy = [ord(x) for x in x]      # 列表表达式内部有自己的局部域
print(x, dumy)

def clock(label, cmd):
    # default_repeat为3,所以会有三个测试值,每次重复执行TIMES次cmd命令.
    res = timeit.repeat(cmd, setup=SETUP, number=TIMES)
    print(label, *('{:.3f}'.format(x) for x in res))

clock('listcomp        :', '[ord(s) for s in symbols if ord(s) > 127]')
clock('listcomp + func :', '[ord(s) for s in symbols if non_ascii(ord(s))]')
clock('filter + lambda :', 'list(filter(lambda c: c > 127, map(ord, symbols)))')
clock('filter + func   :', 'list(filter(non_ascii, map(ord, symbols)))')
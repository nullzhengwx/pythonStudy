import timeit
import bisect
import sys
import queue

from itertools import islice
from operator import itemgetter, attrgetter
import collections

"""according to
https://github.com/fluentpython/example-code/blob/master/02-array-seq/listcomp_speed.py"""
TIMES = 10000

SETUP = """
symbols = '$¢£¥€¤'
def non_ascii(c):
    return c > 127
"""

HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]

ROW_FMT = '{0:2d} @ {1:2d}    {2}{0:<2d}'

class Fib:
    def __init__(self):
        self.prev = 0
        self.curr = 1

    def __iter__(self):
        return self

    def __next__(self):
        value = self.curr
        self.curr += self.prev
        self.prev = value
        return value

class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age

    def __repr__(self):
        return repr((self.name, self.grade, self.age))

def fib():
    prev, curr = 0, 1
    while True:
        yield  curr
        prev, curr = curr, curr + prev

def demo(bisect_fn):
    for needle in reversed(NEEDLES):
        position = bisect_fn(HAYSTACK, needle)
        offset = position * '   |'
        print(ROW_FMT.format(needle, position, offset))

def numeric_compare(x, y):
    return x - y

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

colors = ['black', 'white']
sizes = ['S', 'M', 'L']
tshirts = [(color, size) for color in colors
           for size in sizes]
print(tshirts)

# 生成器表达式,只是[]变成了(),会逐个产出元素.不会一次生成含有所有元素的列表.
for tshirts in ('%s, %s' % (c ,s) for c in colors for s in sizes) :
    print(tshirts)

lax_coordinates = (33.9425, -118.408056)
city, year, pop, chg, area = ('Tokyo', 2003, 32450, 0.66, 8014)
traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'), ('EXP', 'XDA205856')]
for passport in sorted(traveler_ids):
    print('%s/%s' % passport)

for country, _ in traveler_ids:
    print(country)

#f = Fib()
f = fib()
print(list(islice(f, 0, 10)))

if True:
    bisect_fn = bisect.bisect_left
else:
    bisect_fn = bisect.bisect_right

print('DEMO:', bisect_fn.__name__)
print('haystack ->', '  '.join('%2d' % n for n in HAYSTACK))
demo(bisect_fn)

student_tuples = [
    ('john', "A", 15),
    ('jane', "B", 12),
    ('dave', "B", 10),
]

student_objects = [
    Student('john', "A", 15),
    Student('jane', "B", 12),
    Student('dave', "B", 10),
]

print(sorted(student_tuples, key=itemgetter(1, 2), reverse=True))
print(sorted(student_objects, key=attrgetter("grade", "age"), reverse=True))

# The old way using Decorate-Sort-Undecorate
decorated = [(student.grade, i, student) for i, student in enumerate(student_objects)]
decorated.sort()
for grade, i, student in decorated:
    print(student)

# old way in version 2.*
#print(sorted([5,2,4,1,3], cmp=numeric_compare))
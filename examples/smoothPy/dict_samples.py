import sys
import re
import collections
from types import MappingProxyType # just for python 3.3+

WORD_RE = re.compile(r'\w+')

index = {}
""" 用defaultdict来实现
import collections
index = collections.defaultdict(list)
"""

"""
with open(sys.argv[1], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no =  match.start() + 1
            location = (line_no, column_no)
            index.setdefault(word, []).append(location)
            # 用defaultdict来实现
            #index[word].append(location)


# 以字母顺序打印出结果
for word in sorted(index, key=str.upper):
    print(word, index[word])
"""
class StrKeyDict0(dict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def get(self, k, d=None):
        try:
            return  self[k]
        except KeyError:
            return d

    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()

class StrKeyDict(collections.UserDict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, value):
        self.data[str(key)] = value

""" trainning and testing """
d = StrKeyDict([('2', 'two'), ('4', 'four')])
"""
print(d['2'])
print(d[4])
#print(d[1])
print(d.get('2'))
print(d.get(4))
print(d.get(1, 'N/A'))
print(2 in d)
print(1 in d)
"""

d = {1: "A"}
d_proxy = MappingProxyType(d)
print(d_proxy)
print(d_proxy[1])
# d_proxy[2] = "B"
d[2] ='B'
print(d_proxy[2])

str.format()

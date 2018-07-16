try:
    import cPickle as pickle
except:
    import pickle

d = dict(url="index.html", title='黄页', content='首页')
pickle.dumps(d)
print(d)

"""
import os
if __name__ == '__main__':
    print("current Process (%s) start ..." %(os.getpid()))
    pid = os.fork()
    if pid < 0:
        print('error in fork')
    elif pid == 0:
        print('I am child process(%s) and my parent process is (%s)', (os.getpid(), os.getppid()))
    else:
        print('I(%s) created a child process (%s).', (os.getpid(), pid))

"""

from bs4 import BeautifulSoup

html_str = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1"><!-- Elsie --></a>,
<a href="http://example.com/lacie" class="sister" id="link2"><!-- Lacie --></a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""

# soup = BeautifulSoup(html_str, 'lxml', from_encoding='utf-8')
# print(soup.prettify())

i = 0
if not i :
    print("false")
else :
    print("true")
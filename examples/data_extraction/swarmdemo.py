from urllib import request

response = request.urlopen('http://www.zhihu.com')
info = response.info()
print(info)
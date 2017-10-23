import csv
from collections import Counter

from bs4 import BeautifulSoup
import requests
import json
from dateutil.parser import parse

with open("/home/zhenmie/Documents/python/datas/colon_delimited_stock_prices.txt", 'r') as f:
    reader = csv.DictReader(f, delimiter=":")
    for row in reader:
        date = row["date"]
        symbol = row["symbol"]
        closing_price = float(row["closing_price"])
        print(date, symbol, closing_price)

results = [['test1', 'success', 'Monday'],
           ['test2', 'success, kind of', 'Tuseday'],
           ['test3', 'failure, kind of', 'Wednesday'],
           ['test4', 'failure, utter', 'Thursday']]


today_prices = { 'AAPL' : 90.91, 'MSFT' : 12.34, 'FB' : 54.3}
with open("/home/zhenmie/Documents/python/datas/colon_delimited_stock_prices2.txt", "w") as f:
    writer = csv.writer(f, delimiter=",")
    for value in results:
        writer.writerow(value)

"""
html = requests.get("https://www.baidu.com").text
soup = BeautifulSoup(html, 'html5lib')

all_paragraphs = soup.find_all('p')
print(all_paragraphs)
paragraphs_with_ids = [p for p in soup('p') if p.get("id")]
print(paragraphs_with_ids)

spans_inside_divs = [span
                     for div in soup('div')
                     for span in div('span')]
print(spans_inside_divs)

jingdong_html = requests.get("https://search.jd.com/Search?keyword=%E8%80%B3%E6%9C%BA&enc=utf-8&suggest=1.def.0.V06&wq=erj&pvid=5867cee960e34067981cb36fff8bc01f").text
soup = BeautifulSoup(jingdong_html, 'html5lib')
lis = soup('li', 'gl-item')
print(len(lis))
"""

# """ 里面要用 双引号,不然会解析json抛错.
serialized = """{ "title" : "Data Science Book",
                  "author" : "Joel Grus",
                  "publicationYear" : 2014,
                  "topics" : [ "data", "science", "data science"] }"""

# 解析JSON以创建一个Python字典
deserialized = json.loads(serialized)
if 'data science' in deserialized['topics']:
    print(deserialized)

# 在"https://developer.github.com/v3/"看api
endpoint = "https://api.github.com/users/zhenmie365/repos"
repos = json.loads(requests.get(endpoint).text)

dates = [parse(repo["created_at"]) for repo in repos]
print(dates)
weekday_counts = Counter(date.weekday() for date in dates)
print(weekday_counts)

last_5_repositories = sorted(repos, key=lambda r: r['created_at'], reverse=True)[:5]
print(last_5_repositories)
last_5_languages = [repo["language"] for repo in last_5_repositories]
print(last_5_languages)
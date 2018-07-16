import urllib.request as urlrequest
from bs4 import BeautifulSoup

url = 'http://gz.58.com/qzjavakfgongchengshi/pn{}'

with open('/home/zhenmie/Documents/python/datas/58.csv', 'w', encoding='utf8') as outputfile:
    outputfile.write("id#name#gender#age#experient#school#path\n")

    for i in range(10):
        url_visit = url.format(i)

        crawl_content = urlrequest.urlopen(url_visit).read()
        http_content = crawl_content.decode('utf8')
        soup = BeautifulSoup(http_content, 'html.parser')

        all_item_divs = soup.find_all(class_='infocardLi')

        for each_item_div in all_item_divs:

            message_pic_div = each_item_div.find(class_='infocardMessageOne')
            username = message_pic_div.find(class_='infocardName').get_text()

            outputfile.write('{}\n'.format(username))



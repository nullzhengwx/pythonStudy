import urllib.request as urlrequest
import json
import time
import pandas as pd

df = pd.read_csv("/home/zhenmie/Documents/python/datas/top250_f1.csv", sep="#", encoding='utf8')
urlsplit = df.url.str.split('/').apply(pd.Series)
id_list = list(urlsplit[4])
num = 0

with open('/home/zhenmie/Documents/python/datas/top250_f3.csv', 'w', encoding='utf8') as outputfile:
    outputfile.write("num#rank#alt_title#title#pubdate#language#writer#director#cast#movie_duration#year#movie_type#tags#image\n")

    header_key = 'User-Agent'
    header_value = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'

    for id in id_list:
        url_visit = 'https://api.douban.com/v2/movie/{}'.format(id)
        req = urlrequest.Request(url_visit)
        req.add_header(header_key, header_value)
        crawl_content = urlrequest.urlopen(req).read()
        json_content = json.loads(crawl_content.decode('utf-8'))

        rank = json_content['rating']['average']
        alt_title = json_content['alt_title']
        image = json_content['image']
        title = json_content['title']
        pubdate = json_content['attrs']['pubdate']
        language = json_content['attrs']['language']
        try:
            writer = json_content['attrs']['writer']
        except:
            writer = 'None'
        director = json_content['attrs']['director']
        try:
            cast = json_content['attrs']['cast']
        except:
            cast = []
        movie_duration = json_content['attrs']['movie_duration']
        year = json_content['attrs']['year']
        movie_type = json_content['attrs']['movie_type']
        tags = json_content['tags']
        num = num + 1
        outputfile.write("{}#{}#{}#{}#{}#{}#{}#{}#{}#{}#{}#{}#{}#{}\n".format(num, rank, alt_title, title,
                                                                              pubdate, language, writer,
                                                                              director, cast, movie_duration,
                                                                              year, movie_type, tags, image))
        time.sleep(0.01)
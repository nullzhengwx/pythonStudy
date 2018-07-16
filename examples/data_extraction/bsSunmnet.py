# coding:utf-8
import os
import collections

import matplotlib
from bs4 import BeautifulSoup
import pandas as pd
from matplotlib import pyplot as plt

class competition:

    # 获取本文件的绝对路径，用来帮助生成以下相对应文件的绝对路径
    base_path = os.path.dirname(os.path.realpath(__file__))

    def step_0(self):
        path = "/home/zhenmie/Documents/ml/sunmnet/args/datas/data.csv"
        datas = pd.read_csv(path, sep="#")
        statistics = datas.describe(include=['O'])
        print(statistics)
        print(statistics.index)
        print(statistics.loc['unique', 'genre'])


    def step_1(self):
        # 下面的csv路径可以根据环境而不同，可以是相对路径如“../step01/ans01.csv”,也可以是下面那样的绝对路径
        with open(self.base_path + '/step01/ans01.csv', 'w', encoding='utf8') as outputfile:
            # 以下网页路径也可以根据环境而不同，这里用一个循环再加string.format来轮询这些网页。
            # 如果是题目参数就如：[webs/豆瓣电影1.html, webs/豆瓣电影2.html, webs/豆瓣电影3.html, webs/豆瓣电影4.html, webs/豆瓣电影5.html]
            base_url = self.base_path + "/webs/豆瓣电影{}.html"
            for i in range(1, 6):
                url = base_url.format(i)

                # 因为一下是本地离线网页，直接可以open，如果是怕在线网页，可能要用到requests等包。按环境而不同
                soup = BeautifulSoup(open(url))

                all_item_divs = soup.find_all(class_='item')

                for each_item_div in all_item_divs:
                    pic_div = each_item_div.find(class_='pic')
                    num = pic_div.find('em').get_text()  # 排名
                    title = pic_div.find('img')['alt']  # 电影名称

                    # 下面这些字段信息的获取需要一些处理，可以作为扩展分数
                    bd_div = each_item_div.find(class_='bd')
                    infos = bd_div.find('p').get_text().strip().split('\n')
                    infos_1 = infos[0].split('\xa0\xa0\xa0')
                    # 电影导演
                    director = infos_1[0][4:].rstrip('...').rstrip('/').split('/')[0]

                    infos_2 = infos[1].lstrip().split('\xa0/\xa0')
                    year = infos_2[0]  # 上映时间
                    area = infos_2[1]  # 国家/地区

                    # 导入数据到文件
                    outputfile.write('{};{};{};{};{}\n'.format(num, title, director, year, area,))

    def step_2(self):
        datas = pd.read_csv(self.base_path + '/step01/ans01.csv',
                            names=['num', 'title', 'director', 'year', 'area'], sep=";")
        datas['i'] = 1
        l = datas[['year', 'i']].groupby(['year'], as_index=False).sum().sort_values(by='i', ascending=False)
        print("%d,%d"%(l.iat[0, 0],l.iat[0, 1]))

    def step_3(self):
        datas = pd.read_csv(self.base_path + '/step01/ans01.csv',
                            names=['num', 'title', 'director', 'year', 'area'], sep=";")

        area_map = {}

        for index in datas.index:
            a = datas.iat[index, 4]
            areas = a.split(" ")
            for area in areas:
                if(area_map.get(area)):
                    area_map[area] = area_map.get(area) + 1
                else :
                    area_map[area] = 1

        sum_list = [i for i in range(0, len(area_map))]
        key_list = [k for k in area_map.keys()]
        value_list = [v for v in area_map.values()]

        # 后面文件路径用命令“$fc-list :lang=zh”来获得，难点所在.
        myFont_url = self.base_path + "/wqy-microhei.ttc"
        myFont = matplotlib.font_manager.FontProperties(fname=myFont_url)

        plt.ylabel('影片数量', fontproperties=myFont)
        plt.xlabel("出片地区", fontproperties=myFont)
        plt.xticks(sum_list, key_list, fontproperties=myFont)
        plt.bar(left=sum_list, height=value_list, align='center')
        plt.show()

    def step_4(self):
        print("beginning step 4")

        love_map = {}
        crime_map = {}

        base_url = self.base_path + "/webs/豆瓣电影{}.html"
        for i in range(1, 6):
            url = base_url.format(i)

            # 因为一下是本地离线网页，直接可以open，如果是怕在线网页，可能要用到requests等包。按环境而不同
            soup = BeautifulSoup(open(url))

            all_item_divs = soup.find_all(class_='item')

            for each_item_div in all_item_divs:
                bd_div = each_item_div.find(class_='bd')
                infos = bd_div.find('p').get_text().strip().split('\n')
                infos_2 = infos[1].lstrip().split('\xa0/\xa0')
                genre_list = infos_2[2:]  # 电影类型,list,取第一个元素，是一个空格隔开类型名的字符串
                genre = genre_list[0]

                flag = 0
                if "爱情" in genre :
                    flag = 1
                elif "犯罪" in genre:
                    flag = 2
                else :
                    flag = 0
                if not flag:    # 没有这个类型的影片就不获取信息了
                    continue

                pic_div = each_item_div.find(class_='pic')
                title = pic_div.find('img')['alt']  # 电影名称

                star_div = each_item_div.find(class_='star')
                rating_num = star_div.find(class_='rating_num').get_text()  # 评分
                comment_num = star_div.find_all('span')[3].get_text()[:-3]  # 评价数量
                score_num = round(float(rating_num) * float(comment_num), 2)

                if 1 == flag:
                    love_map.setdefault(title, score_num)
                else:
                    crime_map.setdefault(title, score_num)

        love_sort_map = collections.OrderedDict(sorted(love_map.items(), key=lambda p:p[1]))
        crime_sort_map = collections.OrderedDict(sorted(crime_map.items(), key=lambda p: p[1]))

        love_name = []
        love_score = []
        crime_name = []
        crime_score = []
        num_list = [i for i in range(0, 10)]

        for i in range(0, 10):
            love_item = love_sort_map.popitem()
            love_name.append(love_item[0])
            love_score.append(love_item[1])

            crime_item = crime_sort_map.popitem()
            crime_name.append(crime_item[0])
            crime_score.append(crime_item[1])

        myFont_url = self.base_path + "/wqy-microhei.ttc"
        myFont = matplotlib.font_manager.FontProperties(fname=myFont_url)

        plt.figure(1)
        plt.subplot(211)
        plt.ylabel('影片名称', fontproperties=myFont)
        plt.xlabel("热度系数", fontproperties=myFont)
        plt.xticks(num_list, love_name, fontproperties=myFont)
        plt.plot(num_list, love_score, 'bo', num_list, love_score)
        plt.subplot(212)
        plt.ylabel('影片名称', fontproperties=myFont)
        plt.xlabel("热度系数", fontproperties=myFont)
        plt.xticks(num_list, crime_name, fontproperties=myFont)
        plt.plot(num_list, crime_score, 'bo', num_list, crime_score)
        plt.subplots_adjust(hspace=0.5)
        plt.show()


if __name__ == '__main__' :
    competition = competition()
    competition.step_0()
    # competition.step_1()
    # competition.step_2()
    # competition.step_3()
    # competition.step_4()



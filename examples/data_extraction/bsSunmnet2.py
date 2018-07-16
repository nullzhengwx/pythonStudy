# coding:utf-8
import collections
import os

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt


class competition:

    # 获取本文件的绝对路径，用来帮助生成以下相对应文件的绝对路径
    base_path = os.path.dirname(os.path.realpath(__file__))

    def step_1(self):
        path  = self.base_path + '/datas/data.csv'
        inputPath = self.base_path + '/step01/ans01.csv'
        # path = "/home/zhenmie/Documents/ml/sunmnet/args/datas/data.csv"
        # inputPath = "/home/zhenmie/Documents/ml/sunmnet/args/step01/ans01.csv"
        datas = pd.read_csv(path, sep="#")
        statistics = datas.describe(include=['O'])
        # print(statistics)
        # print(statistics.index)
        # print(statistics.loc['unique', 'genre'])
        init_year, year_num, area, area_num = statistics.loc['top', 'init_year'],\
                                              statistics.loc['freq', 'init_year'],\
                                              statistics.loc['top', 'area'],\
                                              statistics.loc['freq', 'area']
        str1 = "%s,%s;%s,%s" % (init_year, year_num, area, area_num)
        # print(str1)
        with open(inputPath, "w", encoding='utf8') as inputfile:
            inputfile.write(str1)

    def step_2(self):
        path = self.base_path + '/datas/data.csv'
        # path = "/home/zhenmie/Documents/ml/sunmnet/args/datas/data.csv"

        datas = pd.read_csv(path,
                            sep="#")

        area_map = {}

        for index in datas.index:
            a = datas.iat[index, 5]
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
        # myFont_url = "/home/zhenmie/Documents/ml/sunmnet/args/wqy-microhei.ttc"
        myFont_url = self.base_path + "/wqy-microhei.ttc"
        myFont = matplotlib.font_manager.FontProperties(fname=myFont_url)

        plt.ylabel('影片数量', fontproperties=myFont)
        plt.xlabel("出片地区", fontproperties=myFont)
        plt.xticks(sum_list, key_list, fontproperties=myFont)
        plt.bar(left=sum_list, height=value_list, align='center')
        plt.show()

    def step_3(self):
        # path = self.base_path + '/datas/data.csv'
        path = "/home/zhenmie/Documents/ml/sunmnet/top250_args/datas/data.csv"

        datas = pd.read_csv(path,
                            sep="#")

        love_map = {}
        crime_map = {}

        for index in datas.index:
            genres = datas.iat[index, 6]

            flag = 0
            if "爱情" in genres:
                flag = 1
            elif "犯罪" in genres:
                flag = 2
            else:
                flag = 0
            if not flag:  # 没有这个类型的影片就不获取信息了
                continue

            score = round(float(datas.iat[index, 7]) * float(datas.iat[index, 8]), 2)
            if flag == 1:
                love_map.setdefault(datas.iat[index, 1],score)
            else :
                crime_map.setdefault(datas.iat[index, 1], score)

        love_sort_map = collections.OrderedDict(sorted(love_map.items(), key=lambda p: p[1]))
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

        # myFont_url = self.base_path + "/wqy-microhei.ttc"
        myFont_url = "/home/zhenmie/Documents/ml/sunmnet/top250_args/wqy-microhei.ttc"
        myFont = matplotlib.font_manager.FontProperties(fname=myFont_url)

        plt.figure(1)
        plt.subplot(211)
        plt.ylabel('热度系数', fontproperties=myFont)
        plt.xlabel("爱情系列影片名称", fontproperties=myFont)
        plt.xticks(num_list, love_name, fontproperties=myFont)
        plt.plot(num_list, love_score, 'bo', num_list, love_score)
        plt.subplot(212)
        plt.ylabel('热度系数', fontproperties=myFont)
        plt.xlabel("犯罪系列影片名称", fontproperties=myFont)
        plt.xticks(num_list, crime_name, fontproperties=myFont)
        plt.plot(num_list, crime_score, 'bo', num_list, crime_score)
        plt.subplots_adjust(hspace=0.5)
        plt.show()


if __name__ == '__main__' :
    competition = competition()
    # competition.step_1()
    # competition.step_2()
    competition.step_3()



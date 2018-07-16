# coding:utf-8
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class competition:

    # 获取本文件的绝对路径，用来帮助生成以下相对应文件的绝对路径
    base_path = os.path.dirname(os.path.realpath(__file__))

    def step_1(self):
        path  = self.base_path + '/datas/WorldCupMatches.csv'
        inputPath = self.base_path + '/step01/ans01.csv'
        # path = "/home/zhenmie/Documents/ml/sunmnet/worldcup_args/datas/WorldCupMatches.csv"
        # inputPath = "/home/zhenmie/Documents/ml/sunmnet/worldcup_args/step01/ans01.csv"
        matches = pd.read_csv(path)

        mat_c = matches['City'].value_counts().reset_index()


        str1 = "%s,%s" % (mat_c['index'][0].strip(), mat_c['City'][0])
        # print(str1)
        with open(inputPath, "w", encoding='utf8') as inputfile:
            inputfile.write(str1)

    def step_2(self):
        path = self.base_path + '/datas/WorldCups.csv'
        # path = "/home/zhenmie/Documents/ml/sunmnet/worldcup_args/datas/WorldCups.csv"

        cups = pd.read_csv(path)

        cou = cups["Winner"].value_counts().reset_index()

        cou_w = cou.copy()
        cou_w.columns = ['country', 'count']
        cou_w['type'] = "WINNER"

        cou_r = cups['Runners-Up'].value_counts().reset_index()
        cou_r.columns = ['country', 'count']
        cou_r['type'] = 'RUNNER - UP'

        cout_t = pd.concat([cou_w, cou_r], axis=0)
        # print(cout_t)

        sns.set_style('darkgrid')
        plt.figure(figsize=(8, 10))
        sns.barplot('count', 'country', data=cout_t, hue='type', palette=['lime', 'r'],
                    linewidth=1, edgecolor='k'*len(cout_t))
        plt.grid(True)
        plt.legend(loc='center right', prop={'size': 14})
        plt.title('World Cup Final results by nation', color='b')
        plt.show()

    def step_3(self):
        path = self.base_path + '/datas/WorldCupMatches.csv'
        # path = "/home/zhenmie/Documents/ml/sunmnet/worldcup_args/datas/WorldCupMatches.csv"

        matches = pd.read_csv(path)

        # 数据处理
        matches['Home Team Name'] = matches["Home Team Name"].replace("Germany FR", "Germany")
        matches['Away Team Name'] = matches["Away Team Name"].replace("Germany FR", "Germany")
        tt_gl_h = matches.groupby("Home Team Name")["Home Team Goals"].sum().reset_index()
        tt_gl_h.columns = ["team", "goals"]

        tt_gl_a = matches.groupby("Away Team Name")["Away Team Goals"].sum().reset_index()
        tt_gl_a.columns = ["team", "goals"]

        total_goals = pd.concat([tt_gl_h, tt_gl_a], axis=0)
        total_goals = total_goals.groupby("team")["goals"].sum().reset_index()
        total_goals = total_goals.sort_values(by="goals", ascending=False)
        total_goals["goals"] = total_goals["goals"].astype(int)

        gh = matches[["Year", "Home Team Goals"]]
        gh.columns = ["year", "goals"]
        gh["type"] = "Home Team Goals"

        ga = matches[["Year", "Away Team Goals"]]
        ga.columns = ["year", "goals"]
        ga["type"] = "Away Team Goals"

        gls = pd.concat([ga, gh], axis=0)

        sns.set_style('darkgrid')
        plt.figure(1)

        plt.subplot(211)
        ax = sns.barplot("goals", "team", data=total_goals[:10], palette="cool",
                         linewidth=1, edgecolor="k" * 10)

        for i, j in enumerate(total_goals["goals"][:10].astype(str)):
            ax.text(.7, i, j, fontsize=8, color="k")

        plt.title("Teams with highest fifa world cup goals", color='b')
        plt.grid(True)

        plt.subplot(212)
        sns.violinplot(gls["year"], gls["goals"],
                       hue=gls["type"], split=True, inner="quart", palette="husl")
        plt.grid(True)
        plt.title("Home and away goals by year", color='b')
        plt.subplots_adjust(hspace=0.3)

        plt.show()


if __name__ == '__main__' :
    competition = competition()
    competition.step_1()
    # competition.step_2()
    # competition.step_3()



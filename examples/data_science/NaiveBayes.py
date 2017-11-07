# -*- coding:utf-8 -*-

import glob
import random
import re
from collections import defaultdict, Counter

import math

def split_data(data, prob):
    """ split data into fractions [prob, 1 - prob] """
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)

def count_words(training_set):
    """ training set consists of pairs (message, is_spam)
    is_spam is list of [spam_count, non_spam_count]"""
    counts = defaultdict(lambda :[0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1

    return counts

def word_probabilities(counts, total_spam, total_non_spams, k=0.5):
    """ turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam_list[0] + k) / (total_spam + 2 * k),
             (spam_list[1] + k) / (total_non_spams + 2 * k))
            for w, spam_list in counts.items()]

def spam_probability(word_probs, message):
    """ 利用这些单词的概率(以及朴素贝叶斯假设)给邮件赋予概率 """
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0

    # 迭代词汇表中的每一个单词
    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # 如果*word*出现在了邮件中
        # 则增加看到它的对数概率
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # 如果*word*没有出现在邮件中
        # 则增加看不到它的对数概率
        # 也就是log(1 - 看到它的概率)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

class NaiveBayesClassifier:
    """ 朴素贝叶斯分类器 """

    def __iter__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        # 对垃圾邮件和非垃圾邮件计数
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # 通过"pipeline"运行训练数据
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts, num_spams,
                                             num_non_spams, self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)

def p_spam_given_word(word_prob):
    """ uses bayes's theorem to compute p(spam | message contains word) """

    # word_prob 是word_probabilities生成的三元素中的一个
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam /(prob_if_spam + prob_if_not_spam)

def drop_final_s(word):
    """ 简单的 词干分析器"""
    return rb.sub("s$", "", word)

""" training and testing"""
path = r"/home/zhenmie/Documents/ml/dataset/emails/*/*"

data = []

# glob.glob会返回每一个与通配路径所匹配的文件名
for fn in glob.glob(path):
    is_spam = "ham" not in fn

    with open(fn, 'r') as file:
        for line in file:
            # 因为文件有编码错误,读行读不了,下面都运行不了.
            if line.startswith("Subject:"):
                # 移除开头的"Subject: ",保留其余内容
                subject = re.sub(r"^Subject: ", "", line).strip()
                data.append((subject, is_spam))

random.seed(0)
train_data, test_data = split_data(data, 0.75)

classifier = NaiveBayesClassifier()
classifier.train(train_data)

# 三个元素(主题, 确实是垃圾邮件, 预测为垃圾邮件的概率)
classified = [(subject, is_spam, classifier.classify(subject))
             for subject, is_spam in test_data]

# 假设spam_probability > 0.5 对应的是预测为垃圾邮件
# 对(actual is_spam, preicted is_spam)的组合计数
counts = Counter((is_spam, spam_probability > 0.5)
                 for _, is_spam, spam_probability in classified)

# 根据spam_probability从最小到最大排序
classified.sort(key=lambda row: row[2])

# 非垃圾邮件被预测为垃圾邮件的最高概率
spammiest_hams = filter(lambda row: not row[1], classified)[-5:]

# 垃圾邮件被预测为垃圾邮件的最低概率
hammiest_spams = filter(lambda row: row[1], classified)[:5]

words = sorted(classifier.word_probs, key=p_spam_given_word)

spammiest_words = words[-5:]
hammiest_words = words[:5]
import collections
from random import choice

Card = collections.namedtuple('Card', ['rank', 'suit'])
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit)
                       for suit in self.suits
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

""" training and testing """
beer_card = Card('7', 'diamonds')
print(beer_card)

deck = FrenchDeck()
print(len(deck))    # 用到__len__

print(deck[0], deck[-1])    # 用到__getitem__

print(choice(deck), choice(deck), choice(deck))

print(deck[:3], deck[12::13])   # 因为__getitem__操作交给了self._card,所有拥有切片功能

"""
for card in deck:       # 仅仅实现了__getitem__方法,就变成了可迭代
    print(card)
"""

print(Card('Q', 'hearts') in deck)  # 没有实现__contains__也没有所谓,in做一次迭代搜索

"""
for card in sorted(deck, key=spades_high):  # 实现了升序函数,结合sorted就可以排序了
    print(card)
"""
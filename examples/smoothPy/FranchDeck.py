import collections
from random import choice
from math import hypot

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

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __str__(self):
        return 'This is a Vector. It contains value x=%r, y=%r.' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)



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

v1 = Vector(2,3)
print(repr(v1))
print(v1)
print(eval(repr(v1)) == v1)
list1 = [Vector(1,4), Vector(6,10)]
print(str(list1))
import numpy as np
import spacy
import wikipedia
import matplotlib.pyplot as plt
from copy import deepcopy

en_nlp = spacy.load('en_core_web_md')

class Tree:
    def __init__(self, cargo, children=None, lleave=None):
        self.cargo = cargo
        self.lleave = lleave
        self.children = children
        self.roots = []
        self.all_ll = []
        self.post_order = []

    def __str__(self):
        return str(self.cargo)

    def l(self):
        if self.children:
            for child in self.children:
                child.l()
                self.all_ll += child.all_ll
        if not self.children:
            self.lleave = self.cargo
            self.all_ll.append(self.lleave)
        else:
            self.lleave = self.children[0].lleave
            self.all_ll.append(self.lleave)

        return self.all_ll

    def post_order_cargo(self):
        if self.children:
            for child in self.children:
                self.post_order += child.post_order_cargo()

        self.post_order.append(self.cargo)
        return self.post_order

    def keyroots(self, current_lleave=False):
        if not current_lleave:
            self.roots.append(self.cargo)
        if self.children:
            self.roots += self.children[0].keyroots(current_lleave=True)
            if len(self.children) > 1:
                for child in range(1, len(self.children)):
                    self.roots += self.children[child].keyroots(current_lleave=False)

        return self.roots


def to_index(post_cargo, array):
    return [post_cargo.index(i) for i in array]


def to_tree(sentence):
    root = sentence[0]
    while root != root.head:
        root = root.head

    def recurse(r):
        kids = []
        if r.children:
            for child in r.children:
                kids.append(recurse(child))

            return Tree(r, kids)
        else:
            return Tree(r)

    return recurse(root)


def remove(a):
    return 5


def insert(a):
    return 5


def update(a, b):

    if a.similarity(b) == 1:
        return 0
    elif a.similarity(b) == 0 or 1./(a.similarity(b)) > 9:
        return 9
    else:
        return abs(1./(a.similarity(b)))

def distance(A, B, insert_cost, remove_cost, update_cost):
    An, Bn = A.post_order_cargo(), B.post_order_cargo()
    Al, Bl = to_index(An, A.l()), to_index(Bn, B.l())
    Ak, Bk = to_index(An, A.keyroots()), to_index(Bn, B.keyroots())
    Ak.sort()
    Bk.sort()
    treedists = np.zeros((len(An), len(Bn)), float)
    def treedist(i, j):

        m = i - Al[i] + 2
        n = j - Bl[j] + 2
        fd = np.zeros((m, n), float)

        ioff = Al[i] - 1
        joff = Bl[j] - 1

        for x in range(1, m):
            fd[x][0] = fd[x-1][0] + remove_cost(An[x+ioff])
        for y in range(1, n):
            fd[0][y] = fd[0][y-1] + insert_cost(Bn[y+joff])

        for x in range(1, m):
            for y in range(1, n):

                if Al[i] == Al[x+ioff] and Bl[j] == Bl[y+joff]:

                    fd[x][y] = min(
                        fd[x-1][y] + remove_cost(An[x+ioff]),
                        fd[x][y-1] + insert_cost(Bn[y+joff]),
                        fd[x-1][y-1] + update_cost(An[x+ioff], Bn[y+joff]),
                    )
                    treedists[x+ioff][y+joff] = fd[x][y]
                else:

                    p = Al[x+ioff]-1-ioff
                    q = Bl[y+joff]-1-joff

                    fd[x][y] = min(
                        fd[x-1][y] + remove_cost(An[x+ioff]),
                        fd[x][y-1] + insert_cost(Bn[y+joff]),
                        fd[p][q] + treedists[x+ioff][y+joff]
                    )


    for i in Ak:
        for j in Bk:
            treedist(i, j)

    return treedists[-1][-1]


ny = en_nlp(wikipedia.page("New York").content[:6000])
chi = en_nlp(wikipedia.page("Chicago").content[:6000])
la = en_nlp(wikipedia.page("Los_Angeles").content[:6000])
ho = en_nlp(wikipedia.page("Houston").content[:6000])
#pho = en_nlp(wikipedia.page("Phoenix").content)


gor = en_nlp(wikipedia.page("Gorilla").content[:6000])
hor = en_nlp(wikipedia.page("Horse").content[:6000])
#rab = en_nlp(wikipedia.page("Rabbit").content[:1800])
tig = en_nlp(wikipedia.page("Tiger").content[:6000])
moo = en_nlp(wikipedia.page("Moose").content[:6000])

articles = [ny, chi, la, ho, gor, hor, tig, moo]
scores = np.zeros((len(articles), len(articles)))
#article_trees = [[to_tree(s) for s in a.sents] for a in articles]


for i in range(len(articles)):
    for j in range(i, len(articles)):
        curr_score = 0
        for k in articles[i].sents:
            for l in articles[j].sents:
                curr_score += distance(to_tree(k), to_tree(l), insert, remove, update)
        print(curr_score)
        scores[(i, j)] = curr_score

print(scores)

scores_T = deepcopy(scores)
np.fill_diagonal(scores_T, 0)
scores_T = scores_T.T
visu_scores = scores + scores_T

plt.matshow(visu_scores)
plt.colorbar()

plt.show()

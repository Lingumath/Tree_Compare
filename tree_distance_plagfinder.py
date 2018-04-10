import numpy as np
import spacy
import matplotlib.pyplot as plt
from copy import deepcopy
from os import walk

en_nlp = spacy.load('en_core_web_md')

class Tree:
    """
    Builds a tree that provides the keyroots() and l() methods as explained in the
    Zhang et al. paper.

    """

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
    """
    Takes two lists, i.e. post order and keyroots list,  that represent each node
    with its value and returns an array with index based representation of array.

    :param post_cargo: must be post order representation of the tree.
    :param array: could be any list that holds nodes represented as values.
    :return: returns array with index based representation.
    """
    post_cargo_dict = {k: v for v, k in enumerate(post_cargo)}
    return [post_cargo_dict[i] for i in array]


def to_tree(sentence):
    """
    Takes a sentence in the spacy document representation and returns a Tree class object.
    """
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


#Some arbitrary distance functions for testing purposes.
def is_noun(tag):
    if tag=="PROPN" or tag=="NOUN" or tag=="PRON":
        return True
    else:
        return False

def remove(a):
    return 1


def insert(a):
    return 1


def update(a, b):
    pos_a = a.pos_
    pos_b = b.pos_

    if pos_a == pos_b or (is_noun(pos_a) and is_noun(pos_b)):
        return (1-round((a.similarity(b)), 4))*2
    else:
        return (1-round((a.similarity(b)), 4))*2

#todo: comments
def distance(A, B, insert_cost, remove_cost, update_cost):
    """
    Implements Zhang et al. tree edit distance algorithm

    :param A: Tree class object (see ln 9)
    :param B: Tree class object
    :param insert_cost: spacy node -> real number
    :param remove_cost: spacy node -> real number
    :param update_cost: spacy node, spacy node -> real number
    :return: distance matrix
    """
    An, Bn = A.post_order_cargo(), B.post_order_cargo()
    Al, Bl = to_index(An, A.l()), to_index(Bn, B.l())
    Ak, Bk = to_index(An, A.keyroots()), to_index(Bn, B.keyroots())
    Ak.sort()
    Bk.sort()
    treedists = np.zeros((len(An), len(Bn)), float)
    def treedist(i, j):
        """
        This part mostly comes from:
        https://github.com/timtadh/zhang-shasha/blob/master/zss/compare.py

        :param i:
        :param j:
        :return:
        """

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


if __name__ == "__main__":
    dirs = walk("poems")
    texts = []
    for d in dirs:
        for f in d[2]:
            texts.append(open("poems/" + f, "r", encoding="utf8").read())


    articles = [en_nlp(t) for t in texts]
    scores = np.zeros((len(articles), len(articles)))
    # article_trees = [[to_tree(s) for s in a.sents] for a in articles]


    for i in range(1):
        for j in range(i, len(articles)):
            all_score = []
            curr_score = 0
            arts_in_i = 0
            arts_in_j = 0
            cnt_j = True
            for k in articles[i].sents:
                arts_in_i += 1
                for l in articles[j].sents:
                    if cnt_j:
                        arts_in_j += 1
                    all_score.append(distance(to_tree(k), to_tree(l), insert, remove, update))
                cnt_j = False

            print(all_score)
            print("------")
            curr_score = sum(sorted(all_score)[:min(arts_in_i, arts_in_j)])

            scores[(i, j)] = curr_score

    print(scores)


    visu_scores = np.expand_dims(scores[0], axis=0)

    plt.matshow(visu_scores)
    plt.colorbar()

    plt.show()

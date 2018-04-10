import numpy as np
import spacy
import wikipedia
import matplotlib.pyplot as plt
from copy import deepcopy

en_nlp = spacy.load('en_core_web_md')
de_nlp = spacy.load('de')

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
def remove(a):
    return 1


def insert(a):
    return 1


def update(a, b):
    if a == b:
        return 0
    else:
        return 1
    #
    # if a == b:
    #     return 0
    # elif a.similarity(b) == 0:
    #     return 9
    # else:
    #     return min(abs(1./(a.similarity(b))), 9)

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

    s0 = de_nlp(u'In einem Land, in dem nur zwei Güter produziert und verbraucht werden, haben Produktion und Verbrauch /'
                u'von Gut X externe Nutzen zur Folge und Produktion und Verbrauch von Gut Y externe Kosten. Mit Blick auf die effiziente/ '
                u'Produktionsmenge: Würde in einem unregulierten Markt zu viel oder zu wenig von Gut X bzw. Gut Y produziert werden?')

    s1 = de_nlp(u'Die nachfolgende Tabelle gibt an, wieviele Tonnen Weizen und Roggen in einem Jahr in Land X und Land Y /'
                u'unter Verwendung derselben Menge an Produktionsfaktoren hergestellt werden können. Wie sollten die /'
                u'Unternehmen des Landes X gemäß der Theorie des komparativen Kostenvorteils vorgehen?')

    s2 = de_nlp(u'Harter Frost hat die Hälfte der heimischen Apfelernte zerstört. Die Verbraucher reagieren auf den/'
                u' steigenden Apfelpreis mit dem vermehrten Kauf von Orangen, so dass für diese mit einem Anstieg der/'
                u'nachgefragten Menge und des Preises zu rechnen ist. Im Grundmodell von Angebot und Nachfrage bedeutet dies eine:')

    s3 = de_nlp(u'In einer Volkswirtschaft, in der hauptsächlich mit Heizöl geheizt wird, werden neue Vorkommen an Erdgas/'
                u'entdeckt, das Heizöl ersetzen und Wärme zu viel geringeren Kosten erzeugen kann. Wie wirkt sich dies/'
                u' wahrscheinlich auf den Marktpreis und die Produktionsmenge von Heizöl aus?')

    s4 = de_nlp(u'In Neustadt-Sonnenbach agiert ein örtliches Eiscremeunternehmen in einem Arbeits- und Produktmarkt mit/'
                u' hohem Wettbewerb. Es kann Arbeitnehmer für 45 Euro am Tag einstellen und Eiswaffeln zu 1,00 Euro pro/'
                u' Stück verkaufen. Die nachfolgende Tabelle zeigt das Verhältnis zwischen der Arbeitnehmeranzahl und/'
                u' der Anzahl verkaufter Eiswaffeln. Wie viele Arbeitnehmer wird das Unternehmen während der gesamten/'
                u' Zeit, in der es tätig ist, anstellen, um den Gewinn zu maximieren bzw. den Verlust zu minimieren?')

    s5 = de_nlp(u'Die beiden einzigen Cola-Hersteller eines Landes (ACola und B-Cola) entscheiden über Preiserhöhungen/'
                u'und –senkungen für ihre Colas. Die nachfolgende Tabelle zeigt die Preisstrategien der Unternehmen und/'
                u'den zu erwartenden Gewinn bzw. Verlust beider Unternehmen in Millionen Euro. Wenn beide Unternehmen /'
                u'davon ausgehen, dass die Mehrzahl der Verbraucher bald keine Cola mehr trinken, sondern auf andere /'
                u'Produkte umsteigen wird, was ist die logische Folge?')

    s6 = de_nlp(u'Ein kleines Land, das in den vergangenen Jahrzehnten eine hohe Inflation zu verzeichnen hatte, /'
                u'beschließt, den Wert seiner Währung dem der Währung eines großen Landes anzugleichen, das in den /'
                u'vergangenen 50 Jahren nur eine äußerst geringe Inflationsrate zu verzeichnen hatte. Das kleine/'
                u' Land profitiert von diesem Schritt, weil')

    s7 = de_nlp(u'Ein Maultier und ein Esel beförderten Lasten von einigen hundert Pfund. Der Esel beklagte sich über /'
                u'die seine und sagte zu dem Maultier: Ich brauche nur hundert Pfund von deiner Last, um meine doppelt/'
                u' so schwer zu machen wie deine. Darauf antwortete das Maultier: Aber, wenn du mir hundert Pfund von/'
                u' deiner Last abgibst, trage ich dreimal so viel wie du. Wie schwer waren sie beladen?')

    s8 = en_nlp(u'In a country where only two goods are produced and consumed, the production and consumption of Good X results in /'
                u'external benefits, while the production and consumption of Good Y results in external costs. Would unregulated /'
                u'markets produce too much or too little of Good X and Good Y, compared to the efficient output levels for these products?')

    s9 = en_nlp(u'The table below shows the tons of rice and corn that can be produced in Country X and Country Y in one/'
                u' year, using the same amount of productive resources. According to the theory of comparative advantage,/'
                u' what should firms in Country X do?')

    s10 = en_nlp(u'A recent hurricane destroyed half of the orange crop. Consumers are responding to an increase in the/'
                u' price of oranges by buying more apples. This change is expected to increase the price and quantity/'
                u' of apples sold. In terms of basic supply and demand analysis, there has been a:')

    s11 = en_nlp(u'In an economy where heating oil is the primary source of heat for most households, new supplies of/'
                u' natural gas, a substitute for heating oil, are discovered. Natural gas provides heat at a much lower/'
                u' cost. What is the most likely effect of these discoveries on the market price and quantity of heating/'
                u' oil produced?')

    s12 = en_nlp(u'In Sunshine City, one local ice cream company operates in a competitive labor market and product/'
                u' market. It can hire workers for $45 a day and sell ice cream cones for $1.00 each. The table below /'
                u'shows the relationship between the number of workers hired and the number of ice cream cones produced/'
                u' and sold. As long as the company stays in business, how many workers will it hire to maximize /'
                u'profits or minimize losses?')

    s13 = en_nlp(u'Suppose the only two cola companies (Acola and Bcola) in a nation are deciding whether to charge high/'
                u' or low prices for their colas. The companies’ price strategies are shown in the table below. The four/'
                u' pairs of payoff values show what each company expects to earn or lose in millions of dollars, /'
                u'depending on what the other company does. If both companies believe that most consumers are soon going/'
                u' to quit drinking colas, and switch to other products, what is the equilibrium outcome?')

    s14 = en_nlp(u'A small country that has experienced high inflation for the past decade decides to set the value of /'
                u'its currency equal to the value of a currency in a large nation that has had very low inflation for /'
                u'the past 50 years. The small country benefits because this action:')

    s15 = en_nlp(u'A mule and an ass were carrying burdens amounting to several hundred weight. The ass complained of/'
                 u' this, and said to the mule, I need only one hundred weight of your load, to make mine twice as heavy/'
                 u' as yours; to which the mule answered, But if you give me a hundred weight of yours, I shall be /'
                 u'loaded three times as much as you will be. How many hundred weight did each carry?')

    docs = [s0, s1, s2, s3, s4, s8, s9, s10, s11, s12]

    A = to_tree(s1)
    B = to_tree(s1)
    d = distance(A, B, insert, remove, update)
    print("Distance:")
    print(d)

    scores = np.zeros((len(docs), len(docs)))

    for a in range(len(docs)):
        for b in range(a, len(docs)):
            scores[(a, b)] = distance(to_tree(docs[a]), to_tree(docs[b]), insert, remove, update)

    scores_T = deepcopy(scores)
    np.fill_diagonal(scores_T, 0)
    scores_T = scores_T.T
    visu_scores = scores + scores_T

    plt.matshow(visu_scores)
    plt.colorbar()

    plt.show()

    ger_v_ger = 0
    eng_v_eng = 0
    ger_v_all = 0
    eng_v_all = 0

    for n, i in enumerate(visu_scores):
        v_ger = sum(i[:8])
        v_eng = sum(i[8:])
        v_all = sum(i)

        print("####################")
        print('#### Sentence ', n, ' ####')
        print(v_ger, " v German")
        print(v_eng, " v English")
        print(v_all, " v All")

        if n < 8:
            ger_v_ger += v_ger
            ger_v_all += v_all
        else:
            eng_v_eng += v_eng
            eng_v_all += v_all

    print("##### G v G #####")
    print(ger_v_ger)
    print("##### E v E #####")
    print(eng_v_eng)
    print("##### G v A #####")
    print(ger_v_all)
    print("##### E v A #####")
    print(eng_v_all)


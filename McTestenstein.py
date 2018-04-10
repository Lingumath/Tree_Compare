import numpy as np
import spacy
from os import walk
# import wikipedia
# import matplotlib.pyplot as plt
# from copy import deepcopy
#
en_nlp = spacy.load('en_core_web_md')
de_nlp = spacy.load('de')

# tokens = en_nlp(u'likes eats hates dislikes loves ')
#
# scores = np.zeros((len(tokens), len(tokens)))
#
# for i, token1 in enumerate(tokens):
#     for j, token2 in enumerate(tokens):
#         print(token1.text, token1.pos_, token2.text, token2.pos_)
#         print(token1.similarity(token2))
        # if token1.tag_ == token2.tag_:
        #     scores[(i, j)] = (1 - round(token1.similarity(token2), 4))**2
        # else:
        #     scores[(i, j)] = (1 - round(token1.similarity(token2), 4))*2


# scores_T = deepcopy(scores)
# np.fill_diagonal(scores_T, 0)
# scores_T = scores_T.T
# visu_scores = scores + scores_T
#
# plt.matshow(visu_scores)
# plt.colorbar()
#
# plt.show()

# for token in tokens:
#     print(token.pos_, token.text)
#     print(" ")





# arti = en_nlp(wikipedia.page("Jim_Carrey").content[:3300])
#
# aritlengs = []
# for i in arti.sents:
#     aritlengs.append(len(i))
#
# ny = en_nlp(wikipedia.page("Blind_Guardian").content[:3300])
# chi = en_nlp(wikipedia.page("Metallica").content[:3300])
# la = en_nlp(wikipedia.page("Slayer").content[:3300])
# ho = en_nlp(wikipedia.page("Opeth").content[:3300])
# # pho = en_nlp(wikipedia.page("Phoenix").content)
#
#
# gor = en_nlp(wikipedia.page("Run_the_Jewels").content[:3300])
# hor = en_nlp(wikipedia.page("Donald_Glover").content[:3300])
# # rab = en_nlp(wikipedia.page("Rabbit").content[:1800])
# tig = en_nlp(wikipedia.page("Kendrick_Lamar").content[:3300])
# moo = en_nlp(wikipedia.page("Tyler,_the_Creator").content[:3300])
#
# articles = [ny, chi, la, ho, gor, hor, tig, moo]

if __name__ == "__main__":
    dirs = walk("cities")
    texts = []
    for d in dirs:
        for f in d[2]:
            texts.append(open("cities/" + f, "r", encoding="utf8").read())


    articles = [de_nlp(t) for t in texts[:10]] + [en_nlp(t) for t in texts[10:]]

s1, s2, s3 = [], [], []

for a in articles:
    alengs = []
    for i in a.sents:
        alengs.append(len(i))
    print("&" + str(round(np.std(alengs), 2))+ "&" + str(round(np.mean(alengs), 2))+"&" + str(round(np.median(alengs), 2)))
    s1.append(np.std(alengs))
    s2.append(np.mean(alengs))
    s3.append(np.median(alengs))

print("German German")
print(round(np.std(s1[:5]), 2),round(np.mean(s1[:5]), 2),round(np.median(s1[:5]), 2))
print("German English")
print(round(np.std(s1[:10]), 2),round(np.mean(s1[:10]), 2),round(np.median(s1[:10]), 2))
print("English German")
print(round(np.std(s1[:15]), 2),round(np.mean(s1[:15]), 2),round(np.median(s1[:15]), 2))
print("English English")
print(round(np.std(s1[15:]), 2),round(np.mean(s1[15:]), 2),round(np.median(s1[15:]), 2))

print("German German")
print(round(np.std(s2[:5]), 2),round(np.mean(s2[:5]), 2),round(np.median(s2[:5]), 2))
print("German English")
print(round(np.std(s2[:10]), 2),round(np.mean(s2[:10]), 2),round(np.median(s2[:10]), 2))
print("English German")
print(round(np.std(s2[:15]), 2),round(np.mean(s2[:15]), 2),round(np.median(s2[:15]), 2))
print("English English")
print(round(np.std(s2[15:]), 2),round(np.mean(s2[15:]), 2),round(np.median(s2[15:]), 2))

print("German German")
print(round(np.std(s3[:5]), 2),round(np.mean(s3[:5]), 2),round(np.median(s3[:5]), 2))
print("German English")
print(round(np.std(s3[:10]), 2),round(np.mean(s3[:10]), 2),round(np.median(s3[:10]), 2))
print("English German")
print(round(np.std(s3[:15]), 2),round(np.mean(s3[:15]), 2),round(np.median(s3[:15]), 2))
print("English English")
print(round(np.std(s3[15:]), 2),round(np.mean(s3[15:]), 2),round(np.median(s3[15:]), 2))

# from os import walk
#
# dirs = walk("taskb")
# texts = []
#
# # a=open("taska/g0pA_taska.txt", encoding="utf8")
# # print(a.read())
#
# for d in dirs:
#     for f in d[2]:
#         print(f)
#         texts.append(open("taskb/"+f, "r", encoding="utf8").read())
#
#
#
# t0 = en_nlp(texts[0])
#
# for s in t0.sents:
#     print(s)
#     print("-----------------------------------------------")



# files = []
# for f in walk("taska"):
#     files.append(open(f).read())

# doc1 = en_nlp(u"The grishnham barked.")
# doc2 = en_nlp(u"The grishnham swam.")
# doc3 = en_nlp(u"the grishnham people live in canada.")
#
# for doc in [doc1, doc2, doc3]:
#     labrador = doc[1]
#     dog = en_nlp(u"dog")
#     print(dog.text)
#     print(labrador.similarity(dog))




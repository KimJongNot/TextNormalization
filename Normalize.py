import gensim
from gensim.models import FastText
from gensim.models import Word2Vec
import textdistance
import ast
import random
import pandas as pd
import math

kbbi = open("data/kbbi.txt").read().splitlines()
kbbi = set(kbbi)
fixWords = pd.read_csv('data/fixWords.csv')
fixWords = dict(zip(fixWords.ragamTB, fixWords.B))
    
class NormalizeText():
    def __init__(self, nsw,t1,t2,t3, modelType):
        '''
        nsw     : non standard word --string
        model   : either word2vec model or fasttext model --.model
        t1      : how many candidates want to generate --int
        t2      : minimun score to determined that two words are similar (only based on cosine similarity)              
                -- int
        t3      : minimun score to determined that two words are similar (jaro-winkler and levenshtein)
                -- int
        '''
        self.nsw = nsw
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.model_ft  = 'C:/anna/Thesis/FastText/fastTextmodelJktnonEnglish_alpha025_window10_epoch300_size300.model'
        self.model_w2v = 'C:/anna/Thesis/Models/Word2VecmodelJktnonEnglish_alpha025_window5_epoch200_size150.model'
        self.modelType = modelType
        if self.modelType == 'ft' or modelType =='1':
            self.model = FastText.load(self.model_ft)
        elif self.modelType == 'wtv' or modelType =='0':
            self.model = Word2Vec.load(self.model_w2v)
        else:
            self.model = FastText.load(self.model_ft)

    def fixWord(self, inKbbiNotBaku):
        global fixWords
        try:
            return [inKbbiNotBaku[0], fixWords[inKbbiNotBaku[1]]]
        except:
            return inKbbiNotBaku


    def findCandidates(self):
        '''
        generate t1 candidates that most similar to the nsw word using the model
        return  : candidates --list of string
        '''
        global kbbi
        try:
            candidates = self.model.wv.most_similar(self.nsw, topn = self.t1)
            candidates = [i for i in candidates if i[0] in kbbi and len(i[0])>1]
        except:
            return []
        return candidates


    def emb_JWdist_LevDist(self, listofwords):
        '''
        Find the most similar word to nsw in listofwords
        listofwords     : candidates -- list of string
        output          : either [0, referenceWord] or [3, standar word]. 3 represents that the output is 
                          received based on jaro winkler and levenshtein similarities
        '''
        if listofwords!=[]:
            JWdist = [ textdistance.jaro_winkler.normalized_similarity(i[0],self.nsw) for i in listofwords ]
            LevDist = [ textdistance.levenshtein.normalized_similarity(i[0],self.nsw) for i in listofwords ]
            distance = [0.5*listofwords[i][1] + 0.25*JWdist[i] + 0.25*LevDist[i] for i in range(len(listofwords))]
            choosenWord = [[listofwords[i][0],distance[i]] for i in range(len(listofwords)) if distance[i] > self.t3]    
            if choosenWord != []:
                ans = max(choosenWord, key=lambda x:x[1])
                return [3,ans[0]]
            else:
                return [0, self.nsw]
        else:
            return [0, self.nsw]


    def modelSim(self, candidates):
        '''
        Find the most similar word to nsw in candidates
        output  : either [2, standar word] or call another function. 
                  2 represents that the output is received based on cosine sim only -- list
        '''
        if candidates[0][1] > self.t2:
            return [2, candidates[0][0]]
        else:
            return self.emb_JWdist_LevDist(candidates)

        
    def normalize(self, candidates):
        '''
        Check wether normalize the nsw or not  
        candidates : candidates taken from findCandidates function
        '''
        global kbbi
        if self.nsw in kbbi or len(self.nsw)==1 :
            return [0,self.nsw]
        if candidates!=[]:
            return self.modelSim(candidates)
        else:
            return [0, self.nsw]


    def NormalizeWord(self):
        '''
        Combine all the functions to get the result
        result  : List of indicator number and the standar word (sw).
        '''
        candidates = self.findCandidates()
        sw_predicted = self.normalize(candidates)
        sw_predicted = self.fixWord(sw_predicted)
        return sw_predicted

if __name__ == '__main__':
    
    typeFile = input("Type '1' to use FastText or '0' to use Word2Vec: " )
    test = NormalizeText('ujan', 30, 0.85, 0.55, typeFile)
    print(test.NormalizeWord())   
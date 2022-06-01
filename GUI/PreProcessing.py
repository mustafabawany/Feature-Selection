import os
import re
import numpy as np
import pandas as pd
from csv import writer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
lemma = WordNetLemmatizer()

class Preprocessing:
    
    def Create_CSV(self):
        
        """Reading Documents in Course Folder"""
        i = 0
        folderPath = os.path.join(os.getcwd() , "/course-cotrain-data/fulltext/course")
        for filename in os.listdir(folderPath):
            with open (os.path.join(folderPath , filename) , 'r' , errors='ignore') as filehandler:
                text = filehandler.read()
                S = BeautifulSoup(text , 'html.parser').text

                #Converting BeautifulSoup object into String
                S = str(S)

                """Lower Casing Words"""
                preProcessed0 = S.lower()
                
                """Removing Encodigs/Numbers"""
                preProcessed1 = self.removeEncodings(preProcessed0)

                """Removing Stop Words"""
                preProcessed2 = self.removeStopWords(preProcessed1)

                """Removing Punctuations"""
                preProcessed3 = self.removePunctuations(preProcessed2)

                """Applying Lemmatization"""
                preProcessed4 = self.lemmatizeSentence(preProcessed3)

                """Creating CSV"""
                list_data = []
                list_data.append(i)
                list_data.append(filename)
                list_data.append(S)
                list_data.append(preProcessed3)
                list_data.append(preProcessed4)
                list_data.append("Course")

                with open('clean_data.csv' , 'a') as fh:
                    writer_obj = writer(fh)
                    writer_obj.writerow(list_data)
                    fh.close()

                i = i+1
  
        folderPath = os.path.join(os.getcwd() , "/course-cotrain-data/fulltext/non-course")
        for filename in os.listdir(folderPath):
            with open (os.path.join(folderPath , filename) , 'r' , errors='ignore') as filehandler:
                text = filehandler.read()
                S = BeautifulSoup(text , 'html.parser').text

                #Converting BeautifulSoup object into String
                S = str(S)

                """Lower Casing Words"""
                preProcessed0 = S.lower()
                
                """Removing Encodigs/Numbers"""
                preProcessed1 = self.removeEncodings(preProcessed0)

                """Removing Stop Words"""
                preProcessed2 = self.removeStopWords(preProcessed1)

                """Removing Punctuations"""
                preProcessed3 = self.removePunctuations(preProcessed2)
                
                """Applying Lemmatization"""
                preProcessed4 = self.lemmatizeSentence(preProcessed3)

                """Creating CSV"""
                list_data = []
                list_data.append(i)
                list_data.append(filename)
                list_data.append(S)
                list_data.append(preProcessed3)
                list_data.append(preProcessed4)
                list_data.append("No Course")
                
                with open('clean_data.csv' , 'a') as fh:
                    writer_obj = writer(fh)
                    writer_obj.writerow(list_data)
                    fh.close()

                i = i+1

    def stemSentence(self, sentence):
        token_words = word_tokenize(sentence)
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(ps.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def removeStopWords(self, sentence):
        resultSentence = sentence.split()
        resultSentence = [word for word in resultSentence if not word in stopwords.words('english')]
        return ' '.join(resultSentence)

    def removePunctuations(self, sentence):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        newSentence = ""
        for word in sentence:
            if (word in punctuations):
                newSentence = newSentence + " "
            else: 
                newSentence = newSentence + word
        return newSentence
    
    def removeEncodings(self , sentence):
        return re.sub('[^a-zA-Z]'," ",str(sentence))

    def lemmatizeSentence(self , sentence):
        token_words = word_tokenize(sentence)
        lemma_sentence=[]
        for word in token_words:
            lemma_sentence.append(lemma.lemmatize(word))
            lemma_sentence.append(" ")
        return "".join(lemma_sentence)

# if __name__ == "__main__":

#     Object = Preprocessing()
    
#     """Reading Documents And Creating CSV"""

#     Object.Create_CSV()
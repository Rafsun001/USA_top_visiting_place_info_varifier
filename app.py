from flask import Flask,render_template,request
import numpy as np

### must install while deploying ###
#pip install openpyxl


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import string
import re
from bs4 import BeautifulSoup

from difflib import SequenceMatcher
import pickle
import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def verifier_ui():
    return render_template('index.html')

df = pd.read_excel("USA Information main copy.xlsx")
def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('(', '')
    q = q.replace(')', '')
    q = q.replace('-', ' ')
    
    
    
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    

    
    return q
df['Information'] = df['Information'].apply(preprocess)

for i in df['Information']:
    translator = str.maketrans('', '', string.punctuation)
    clean_sentence = i.translate(translator)
    df=df.replace(i,clean_sentence)

text_data = df['Information'].tolist()

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the text data into a document-term matrix
X = vectorizer.fit_transform(text_data)

X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

df = pd.concat([df, X_df], axis=1)

@app.route('/usa_info_vari',methods=['post'])
def recommend():
    
    input_dt = request.form.get('user_input')
    def preprocess(q):
    
        q = str(q).lower().strip()
        
        # Replace certain special characters with their string equivalents
        q = q.replace('%', ' percent')
        q = q.replace('$', ' dollar ')
        q = q.replace('₹', ' rupee ')
        q = q.replace('€', ' euro ')
        q = q.replace('@', ' at ')
        q = q.replace('(', '')
        q = q.replace(')', '')
        q = q.replace('-', ' ')
        
        
        
        
        # Decontracting words
        # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
        # https://stackoverflow.com/a/19794953
        contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }

        q_decontracted = []

        for word in q.split():
            if word in contractions:
                word = contractions[word]

            q_decontracted.append(word)

        q = ' '.join(q_decontracted)
        q = q.replace("'ve", " have")
        q = q.replace("n't", " not")
        q = q.replace("'re", " are")
        q = q.replace("'ll", " will")
        
        # Removing HTML tags
        q = BeautifulSoup(q)
        q = q.get_text()
        
        return q
    
    
    data=[]
    #doing preprocess
    v=preprocess(input_dt)

    #Number of sentence contain
    sentences = sent_tokenize(v)
    data.append(len(sentences))

    #removing puncuatin marks
    translator = str.maketrans('', '', string.punctuation)
    clean_sentence = v.translate(translator)

    #count of number of numerical value
    number_pattern = r'\d+'
    matches = re.findall(number_pattern, clean_sentence)
    numerical_count = len(matches)
    data.append(numerical_count)

    #count verb, adverd, adjective etc
    tokens = nltk.word_tokenize(clean_sentence)
    pos_tags = nltk.pos_tag(tokens)
    verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
    data.append(verb_count)
    adverb_count = sum(1 for word, pos in pos_tags if pos.startswith('RB'))
    data.append(adverb_count)
    noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
    data.append(noun_count)
    adjective_count = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
    data.append(noun_count)
    conjunction_count = sum(1 for word, pos in pos_tags if pos.startswith('CC'))
    data.append(conjunction_count)
    preposition_count = sum(1 for word, pos in pos_tags if pos.startswith('IN'))
    data.append(noun_count)
    interjection_count = sum(1 for word, pos in pos_tags if pos.startswith('UH'))
    data.append(noun_count)
    pronoun_tags = ('PRP', 'PRP$', 'WP', 'WP$')
    pronoun_count = sum(1 for word, pos in pos_tags if pos in pronoun_tags)
    data.append(pronoun_count)

    # length of sentence
    data.append(len(clean_sentence))

    #Number of words
    words = clean_sentence.split()
    data.append(len(words))

    #most verb count
    vword=[]
    tokens = nltk.word_tokenize(clean_sentence)
    pos_tags = list(nltk.pos_tag(tokens))
    for j in pos_tags:
        c=list(j)
        if c[1]=="VB":
            vword.append(c[0])
        else:
            pass
    nd={}   
    for i in range(len(vword)):
        xn=0
        for j in vword:
            if vword[i]==j:
                xn=xn+1
            else:
                pass
        nd[vword[i]]=xn
        xn=0
    nd = {k: v for k, v in sorted(nd.items(), key=lambda item: item[1])}       
    vword=list(nd.keys())
    if len(vword)>=2 or len(vword)==1:
        vw1=0
        for k in vword:
            if k==vword[len(vword)-1]:
                vw1=vw1+1
            else:
                pass
        data.append(vw1)
        vw1=0
    elif len(vword)==0:
        data.append(0)

    else:
        pass

    #Most adverb count
    vword=[]
    tokens = nltk.word_tokenize(clean_sentence)
    pos_tags = list(nltk.pos_tag(tokens))
    for j in pos_tags:
        c=list(j)
        if c[1]=="RB":
            vword.append(c[0])
        else:
            pass
    nd={}   
    for i in range(len(vword)):
        xn=0
        for j in vword:
            if vword[i]==j:
                xn=xn+1
            else:
                pass
        nd[vword[i]]=xn
        xn=0
    nd = {k: v for k, v in sorted(nd.items(), key=lambda item: item[1])}       
    vword=list(nd.keys())
    if len(vword)>=2 or len(vword)==1:
        vw1=0
        for k in vword:
            if k==vword[len(vword)-1]:
                vw1=vw1+1
            else:
                pass
        data.append(vw1)
        vw1=0
    elif len(vword)==0:
        data.append(0)

    else:
        pass


    #Count of year
    years = re.findall(r'\b\d{4}\b', clean_sentence)
    vx=len(years)
    data.append(vx)

    #total number of stopwrods
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(clean_sentence)
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    data.append(stopword_count)
    
    
    print(len(data))
    # Assuming `clean_sentence` is a list of strings or a single string
    q2_bow = vectorizer.transform([clean_sentence]).toarray()
    v=np.hstack((np.array(data).reshape(1,16),q2_bow))
    prediction = model.predict(v)
    if prediction==1:
        return render_template('index.html',prediction_text="The given information is correct")
    else:
        return render_template('index.html',prediction_text="The given information is wrong")
        

if __name__ == '__main__':
    app.run()
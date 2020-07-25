#!/usr/bin/python
# --------------imports--------------------
import re
import string
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from contextlib import closing
from bs4 import BeautifulSoup
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
# -----------------x----------------------------

# ------------main function------------------------
def main():
    def url_to_content(url_files):
        """
            here instead of sending a get request to url i have saved the file locally and use
            that file to scrape the data. This is done beacuse i want to run the code continiously
            for debuging purpose and doing so using the get request would considered as spaming and
            websites server would tend to block my ip as spammerself.    

            input -> url_files
            output-> text
            usecase -> parse html to text
        """
        page = open(url_files).read()
        soup = BeautifulSoup(page,"html.parser")
        text = [p.text for p in soup.find_all('p')]
        return text

    # list of locally stored html files
    url_files = ['aus.html','england.html','japan.html','nz.html','usa.html']
    #  list of countries 
    country = ['Australia','England','Japan','NewZealand','USA']

    # parsing html to transcript
    transcripts = [ url_to_content(u) for u in url_files]

    data = {}
    for i,c in enumerate(country):
        data[c] = transcripts[i]



    def combine_text(list_of_text):
        '''Takes a list of text and combines them into one large chunk of text.'''
        combined_text = ' '.join(list_of_text)
        return combined_text

    data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

    pd.set_option('max_colwidth',200)

    data_df = pd.DataFrame.from_dict(data_combined).transpose()
    data_df.columns = ['transcript']
    data_df = data_df.sort_index()

    full_names = ['Scott Morrison','Boris Johnson','Shinzo Abe','Jacinda Ardern','Donald Trump']
    data_df['full_name'] = full_names



    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    round1 = lambda x: clean_text_round1(x)
    data_clean = pd.DataFrame(data_df.transcript.apply(round1))
    # print(data_clean)

    def clean_text_round2(text):
        '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\t', '', text)
        text = re.sub('–', '', text)
        text = re.sub('—', '', text)
        text = re.sub('we use cookies to collect information about how you use govuk we use this information to make the website work as well as possible and improve government services youve accepted all cookies you can change your cookie settings at any time prime minister boris johnson made a statement on coronavirus', '', text)
        text = re.sub('dont include personal or financial information like your national insurance number or credit card details to help us improve govuk wed like to know more about your visit today well send you a link to a feedback form it will take only  minutes to fill in dont worry we wont send you spam or share your email address with anyone                     open government licence                   all content is available under the open government licence  except where otherwise stated', '', text)
        text = re.sub('skip to main content please activate javascript function on your browser   home  news  speeches and statements by the prime minister  june   press conference by prime minister shinzo abe june', '', text)
        text = re.sub('this is a modal window', '', text)
        text = re.sub('beginning of dialog window escape will cancel and close the window end of dialog window  this modal can be closed by pressing the escape key or activating the close button  watch pm jacinda arderns full speech  credits newshub  prime minister jacinda arderns full  speech', '', text)
        text = re.sub('President Donald Trump', '', text)
    
        return text

    round2 = lambda x: clean_text_round2(x)

    data_clean = pd.DataFrame(data_clean.transcript.apply(round2))

    data_clean['full_name'] = full_names
    # print(data_clean)

    data_df.to_pickle("corpus.pkl")
    data_clean.to_pickle("corpus_clean.pkl")
    # exit()


    cv = CountVectorizer()
    data_cv = cv.fit_transform(data_clean.transcript)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean.index
    # print(data_dtm)


    data = data_dtm.transpose()
    # print(data.head())
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(40)
        top_dict[c]= list(zip(top.index, top.values))

    # print(top_dict)

    for country, top_words in top_dict.items():
        print(country)
        print(', '.join([word for word, count in top_words[0:40]]))
        print('---')




    words = []
    for c in data.columns:
        top = [word for (word, count) in top_dict[c]]
        for t in top:
            words.append(t)
            
    # print(words)
    # print(Counter(words).most_common())
    add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
    # print(add_stop_words)
    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    cv = CountVectorizer(stop_words=stop_words)
    data_cv = cv.fit_transform(data_clean.transcript)
    data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.index = data_clean.index
    # print(data_stop)

    data = data_stop.transpose()
    # print(data.head())
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(40)
        top_dict[c]= list(zip(top.index, top.values))
    # print(top_dict)

    # for country, top_words in top_dict.items():
    #     print(country)
    #     print(', '.join([word for word, count in top_words[0:40]]))
    #     print('---')

    # word could data matrix  
    wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2", max_font_size=200, random_state=42)

    plt.rcParams['figure.figsize'] = [20,10]

    full_names = ['Scott Morrison','Boris Johnson','Shinzo Abe','Jacinda Ardern','Donald Trump']

    for index, leader in enumerate(data.columns):
        wc.generate(data_clean.transcript[leader])
        
        plt.subplot(3, 4, index+1)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(full_names[index], fontdict={'size':10})
    plt.show()

    unique_list = []
    for leader in data.columns:
        uniques = data[leader].to_numpy().nonzero()[0].size
        unique_list.append(uniques)

    data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['leader', 'unique_words'])

    total_list = []

    for leader in data.columns:
        totals = sum(data[leader])
        total_list.append(totals)
    data_words['total_words'] = total_list
    data_unique_sort = data_words.sort_values(by='unique_words')
    print(data_unique_sort)

    data = pd.read_pickle('corpus.pkl')
    print(data)


    pol = lambda x: TextBlob(x).sentiment.polarity
    sub = lambda x: TextBlob(x).sentiment.subjectivity

    data['polarity'] = data['transcript'].apply(pol)
    data['subjectivity'] = data['transcript'].apply(sub)


    plt.rcParams['figure.figsize'] = [20, 10]

    for index, leader in enumerate(data.index):
        x = data.polarity.loc[leader]
        y = data.subjectivity.loc[leader]
        plt.scatter(x, y, color='blue')
        plt.text(x+.001, y+.001, data['full_name'][index], fontsize=8)
        plt.xlim(-.01, .19) 
        
    plt.title('Sentiment Analysis', fontsize=20)
    plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
    plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

    plt.show()

    def split_text(text, n=15):
        '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

        length = len(text)
        size = math.floor(length / n)
        start = np.arange(0, length, size)
        
        split_list = []
        for piece in range(n):
            split_list.append(text[start[piece]:start[piece]+size])
        return split_list

    list_pieces = []

    for t in data.transcript:
        split = split_text(t)
        list_pieces.append(split)
        
    polarity_transcript = []

    for lp in list_pieces:
        polarity_piece = []
        for p in lp:
            polarity_piece.append(TextBlob(p).sentiment.polarity)
        polarity_transcript.append(polarity_piece)
        

    plt.rcParams['figure.figsize'] = [20, 10]

    for index, leader in enumerate(data.index):    
        plt.subplot(3, 4, index+1)
        plt.plot(polarity_transcript[index])
        plt.plot(np.arange(0,20), np.zeros(20))
        plt.title(data['full_name'][index])
        plt.ylim(ymin=-.3, ymax=.5)
    plt.show()


if __name__ == '__main__':
    main()
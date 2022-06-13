import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import time
from keras.models import load_model
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import lxml
import spacy
import spacy.cli

# Deep learning part
model = load_model('apparel_reviews_200_test1.h5')

with open('tokenizer_apparel_reviews_200_test1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# list of stopwords meaningful in a sentiment analysis
exclude_stopw = [
'what', 'which', 'who', 'whom', 'this', 'that', 'be', 'been', 'being', 'have',
'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
'so', 'than', 'too', 'very', 'can', 'don', "don't", 'should', "should've", 'ain',
'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
'wouldn', "wouldn't"
]

nltk.download('stopwords')

nltk_stopwords = stopwords.words('english')

# exclude meaningful words from a default nltk_stopwords
english_stopwords = [word for word in nltk_stopwords if word not in exclude_stopw]

# stemming
stemmer = SnowballStemmer('english')

# regex for mention and links
regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(content, stem=False):
    content = re.sub(regex, ' ', str(content).lower()).strip()
    tokens = []
    for token in content.split():
        if token not in english_stopwords:
            tokens.append(stemmer.stem(token))
    return " ".join(tokens)


def predict_sentiment(word):

    test_word = preprocess(word)
    tw = tokenizer.texts_to_sequences([test_word])
    tw = pad_sequences(tw,maxlen=100,padding='post')
    prediction = model.predict(tw).item()
    return prediction

# Scraping part

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")

driver = webdriver.Chrome(options=options)

def get_reviews(url):
    
    driver.get(url)
    time.sleep(5)

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'lxml')
    reviews = soup.find_all('span', {'class': 'a-size-base review-text review-text-content'})
    date = soup.find_all('span', {'class': 'a-size-base a-color-secondary review-date'})
    time.sleep(1)

    review_dict = {'sentiment': [], 'review' : [], 'date': [], 'sentiment_score': []}
    for w in reviews:
        target_str = w.text.strip()
        review_dict['review'].append(target_str)
        sent = predict_sentiment(target_str)
        review_dict['sentiment_score'].append(sent)
        if sent < 0.5:
            review_dict['sentiment'].append('negative')
        else:
            review_dict['sentiment'].append('positive')
    for c in date[2:]:
        temp = c.text.strip()
        date_final = temp.replace('Reviewed in the United States on ', '')
        review_dict['date'].append(date_final)

    data = pd.DataFrame(review_dict)
    return data

# Pos Classification part

#download and load pretrained pos tagger
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def pos_spacy(review):
    line = [(w, w.pos_) for w in nlp(review)]
    return line

def pos_value_count(df_column):
    pos_dict = {"noun": [], "verb": [], "adjective": [], "adverb": []}
    for tup in df_column:
        for word, pos in tup:
            if word.is_stop == False:
                if pos == 'NOUN':
                    lem = word.lemma_
                    pos_dict['noun'].append(lem)
                elif pos == 'VERB':
                    lem = word.lemma_
                    pos_dict['verb'].append(lem)
                elif pos == 'ADJ':
                    lem = word.lemma_
                    pos_dict['adjective'].append(lem)
                elif pos == 'ADV':
                    lem = word.lemma_
                    pos_dict['adverb'].append(lem)

    noun_df = pd.DataFrame({'word' : pos_dict['noun']}).value_counts().reset_index()
    noun_df = noun_df.rename({"word": "word", 0 : "count"}, axis=1)
    
    verb_df = pd.DataFrame({'word' : pos_dict['verb']}).value_counts().reset_index()
    verb_df = verb_df.rename({"word": "word", 0 : "count"}, axis=1)
    
    adj_df = pd.DataFrame({'word' : pos_dict['adjective']}).value_counts().reset_index()
    adj_df = adj_df.rename({"word": "word", 0 : "count"}, axis=1)
    
    adv_df = pd.DataFrame({'word' : pos_dict['adverb']}).value_counts().reset_index()
    adv_df = adv_df.rename({"word": "word", 0 : "count"}, axis=1)

    return noun_df, verb_df, adj_df, adv_df

def word_plot(noun_df, adj_df, adv_df, verb_df):
    sns.set(font_scale=1.4)
    if (len(adv_df) > 0) & (len(adj_df) > 0) & (len(verb_df) > 0):
        fig, axs = plt.subplots(1,4, figsize=(18, 25))
        sns.barplot(data=noun_df, x='count', y='word', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=verb_df, x='count', y='word', ax=axs[1])
        axs[1].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
        sns.barplot(data=adj_df, x='count', y='word', ax=axs[2])
        axs[2].set(title='Adjective Word Frequency in Reviews', xlabel='Count', ylabel='Adjective Words in Reviews')
        sns.barplot(data=adv_df, x='count', y='word', ax=axs[3])
        axs[3].set(title='Adverb Word Frequency in Reviews', xlabel='Count', ylabel='Adverb Words in Reviews')
        
    elif (len(adv_df) > 0) & (len(adj_df) == 0) & (len(verb_df) > 0):
        fig, axs = plt.subplots(1,3, figsize=(18, 25))
        sns.barplot(data=noun_df, x='count', y='word', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=verb_df, x='count', y='word', ax=axs[1])
        axs[1].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
        sns.barplot(data=adv_df, x='count', y='word', ax=axs[2])
        axs[2].set(title='Adverb Word Frequency in Reviews', xlabel='Count', ylabel='Adverb Words in Reviews')
        
    elif (len(adv_df) == 0) & (len(adj_df) == 0) & (len(verb_df) > 0):
        fig, axs = plt.subplots(1, 2, figsize=(18, 25))
        sns.barplot(data=noun_df, x='count', y='word', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=verb_df, x='count', y='word', ax=axs[1])
        axs[1].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
    
    elif (len(adv_df) == 0) & (len(adj_df) > 0) & (len(verb_df) > 0):
        fig, axs = plt.subplots(1, 3, figsize=(18, 25))
        sns.barplot(data=noun_df, x='count', y='word', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=verb_df, x='count', y='word', ax=axs[1])
        axs[1].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
        sns.barplot(data=adj_df, x='count', y='word', ax=axs[2])
        axs[2].set(title='Adjective Word Frequency in Reviews', xlabel='Count', ylabel='Adjective Words in Reviews')
            
    elif (len(adv_df) == 0) & (len(adj_df) == 0) & (len(verb_df) == 0):
        fig, axs = plt.subplots(figsize=(18, 25))
        sns.barplot(data=noun_df, x='count', y='word', ax=axs)
        axs.set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
    
    elif (len(noun_df) == 0) & (len(adv_df) == 0) & (len(adj_df) == 0) & (len(verb_df) == 0):
        pass
    
    fig.tight_layout()
    return fig

def positive_plot(df):
    w_posi = df.loc[df['sentiment'] == 'positive']
    if len(w_posi) == 0:
        st.write("There is no positive reviews on the product in recent 9 reviews")
    else:
        # POS tags for each word
        w_posi['review_pos'] = w_posi['review'].apply(lambda x: pos_spacy(x))
        noun_df, verb_df, adj_df, adv_df = pos_value_count(w_posi['review_pos'])
        st.pyplot(word_plot(noun_df, adj_df, adv_df, verb_df))

def negative_plot(df):
    w_neg = df.loc[df['sentiment'] == 'negative']
    if len(w_neg) == 0:
        st.write("There is no negative reviews on the product in recent 9 reviews")
    else:
        # POS tags for each word
        w_neg['review_pos'] = w_neg['review'].apply(lambda x: pos_spacy(x))
        noun_df, verb_df, adj_df, adv_df = pos_value_count(w_neg['review_pos'])
        st.pyplot(word_plot(noun_df, adj_df, adv_df, verb_df))


# Streamlit part
st.title("Real-Time Customer Sentiment Analyzer: Clothing Products in Amazon.com")
st.markdown("#### (Sentiment Analysis Done by a GRU Model, Deep Learning)")
st.markdown("This app collects the most recent customer reviews on a few products and yields sentiment of each review using a GRU neural network model trainned with 1467106 apperal reviews in Amazon.com. It also then dissembles each review by part of speech and displays the frequency of important words to learn what made customer feel negative as well as positive about the product. Since this app is only for a learning and experiment purpose, it scrapes and analyzes only a few Amazon product pages real-time.")
st.markdown("**However, any further development is possible including a sentiment search, a real-time product sentiment analyzer and a dashboard for all products in any online stores.** &nbsp;")
st.markdown("*Each loading takes about 10 seconds, but it can be much faster than that if adjusted for a real application purpose.* &nbsp;")

st.markdown("### Sample: Recent Clothing Review Sentiment in Amazon.com")

url = "https://www.amazon.com/Bsubseach-Sleeve-Blouses-Bathing-Swimwear/product-reviews/B087CNRPF9/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"

ran_df = get_reviews(url)
st.dataframe(ran_df)
st.markdown("&nbsp;")

st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
positive_plot(ran_df)
st.markdown("&nbsp;")

st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
negative_plot(ran_df)
st.markdown("&nbsp;")

st.markdown("### Recent Clothing Review Sentiment by Department")
option = st.selectbox("Choose Department", ("Select One", "Women's Clothing", "Men's Clothing", "Girls' Clothing", "Boys' Clothing"))

if option == "Women's Clothing":
    w_url = "https://www.amazon.com/Levis-Womens-501-Original-Shorts/product-reviews/B09RQTMLXP/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    w_df = get_reviews(w_url)
    st.dataframe(w_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(w_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(w_df)

if option == "Men's Clothing":
    m_url = "https://www.amazon.com/Fruit-Loom-Eversoft-Cotton-T-Shirt/product-reviews/B0B2WV23RL/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    m_df = get_reviews(m_url)
    st.dataframe(m_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(m_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(m_df)

if option == "Girls' Clothing":
    s_url = "https://www.amazon.com/The-Childrens-Place-Girls-Leggings/product-reviews/B09XWRT6RK/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    s_df = get_reviews(s_url)
    st.dataframe(s_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(s_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(s_df)

if option == "Boys' Clothing":
    b_url = "https://www.amazon.com/Marvel-Avengers-Athletic-Shorts-Packs/product-reviews/B072WC6Y1G/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    b_df = get_reviews(b_url)
    st.dataframe(b_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(b_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(b_df)

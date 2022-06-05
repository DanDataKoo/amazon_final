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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, pos_tag_sents
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import lxml

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

# lemmatizer for testing only
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# lemmatizer = WordNetLemmatizer()

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
            # tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)


def predict_sentiment(word):

    test_word = preprocess(word)
    tw = tokenizer.texts_to_sequences([test_word])
    tw = pad_sequences(tw,maxlen=100,padding='post')
    prediction = model.predict(tw).item()
    return prediction

# Scraping part

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
#driver = webdriver.Chrome(options=options)

def get_reviews(url):
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(2)

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'lxml')
    reviews = soup.find_all('span', {'class': 'a-size-base review-text review-text-content'})
    date = soup.find_all('span', {'class': 'a-size-base a-color-secondary review-date'})
    time.sleep(1)

    review_dict = {'review' : [], 'date': [], 'sentiment': [], 'sentiment_score': []}
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

def clean_text(content):
    regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    nltk_stopwords = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    content = re.sub(regex, ' ', str(content).lower()).strip()
    tokens = []
    for token in content.split():
        if token not in nltk_stopwords:
            # tokens.append(stemmer.stem(token))
            tokens.append(token)
    content = " ".join(tokens)
    return content

# data['review_cleaned'] = data['review'].apply(lambda x: clean_text(x))
# data['review_pos'] = pos_tag_sents(data['review_cleaned'].apply(word_tokenize).tolist())


def pos_value_count(column):
    lemmatizer = WordNetLemmatizer()
    pos_list = ['JJ', 'JJR', 'JJS', 'RBR', 'RBS', 'NN', 'NNS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

    noun_dict = {'Noun': []}
    adj_dict = {'Adjective': []}
    adv_dict = {'Comp_Adverb': []}
    verb_dict = {'Verb': []}
    for line in column.tolist():
        for word, pos in line:
            if pos in pos_list:
                if (pos == 'NN') or (pos == 'NNS'):
                    noun_dict['Noun'].append(lemmatizer.lemmatize(word))
                elif (pos == 'JJ') or (pos == 'JJR') or (pos == 'JJS'):
                    adj_dict['Adjective'].append(lemmatizer.lemmatize(word))
                elif (pos == 'RBR') or (pos == 'RBS'):
                    adv_dict['Comp_Adverb'].append(lemmatizer.lemmatize(word))
                else:
                    verb_dict['Verb'].append(lemmatizer.lemmatize(word))

    noun_df = pd.DataFrame(noun_dict)
    noun_df = noun_df['Noun'].value_counts().reset_index()

    adj_df = pd.DataFrame(adj_dict)
    adj_df = adj_df['Adjective'].value_counts().reset_index()

    adv_df = pd.DataFrame(adv_dict)
    adv_df = adv_df['Comp_Adverb'].value_counts().reset_index()

    verb_df = pd.DataFrame(verb_dict)
    verb_df = verb_df['Verb'].value_counts().reset_index()

    return noun_df, adj_df, adv_df, verb_df

def word_plot(noun_df, adj_df, adv_df, verb_df):
    sns.set(font_scale=1.8)
    if len(adv_df) > 0:
        fig, axs = plt.subplots(1,4, figsize=(18, 25))
        sns.barplot(data=noun_df, x='Noun', y='index', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=adj_df, x='Adjective', y='index', ax=axs[1])
        axs[1].set(title='Adjective Word Frequency in Reviews', xlabel='Count', ylabel='Adjective Words in Reviews')
        sns.barplot(data=adv_df, x='Comp_Adverb', y='index', ax=axs[2])
        axs[2].set(title='Adverb Word Frequency in Reviews', xlabel='Count', ylabel='Adverb Words in Reviews')
        sns.barplot(data=verb_df, x='Verb', y='index', ax=axs[3])
        axs[3].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 25))
        sns.barplot(data=noun_df, x='Noun', y='index', ax=axs[0])
        axs[0].set(title='Noun Word Frequency in Reviews', xlabel='Count', ylabel='Noun Words in Reviews')
        sns.barplot(data=adj_df, x='Adjective', y='index', ax=axs[1])
        axs[1].set(title='Adjective Word Frequency in Reviews', xlabel='Count', ylabel='Adjective Words in Reviews')
        sns.barplot(data=verb_df, x='Verb', y='index', ax=axs[2])
        axs[2].set(title='Verb Word Frequency in Reviews', xlabel='Count', ylabel='Verb Words in Reviews')
        #fig.delaxes(axs[1][1])
    fig.tight_layout()
    return fig

def positive_plot(df):
    w_posi = df.loc[df['sentiment'] == 'positive']
    w_posi['review_cleaned'] = w_posi['review'].apply(lambda x: clean_text(x))
    # POS tags for each word
    w_posi['review_pos'] = pos_tag_sents(w_posi['review_cleaned'].apply(word_tokenize).tolist())
    noun_df, adj_df, adv_df, verb_df = pos_value_count(w_posi['review_pos'])
    st.pyplot(word_plot(noun_df, adj_df, adv_df, verb_df))

def negative_plot(df):
    w_neg = df.loc[df['sentiment'] == 'negative']
    w_neg['review_cleaned'] = w_neg['review'].apply(lambda x: clean_text(x))
    # POS tags for each word
    w_neg['review_pos'] = pos_tag_sents(w_neg['review_cleaned'].apply(word_tokenize).tolist())
    noun_df, adj_df, adv_df, verb_df = pos_value_count(w_neg['review_pos'])
    st.pyplot(word_plot(noun_df, adj_df, adv_df, verb_df))


# Streamlit part
st.title("Most Recent Customer Sentiment: Clothing Products in Amazon.com")
st.markdown("#### (Sentiment Analysis Done by a GRU Model, Deep Learning)")
st.markdown("This app collects the most recent customer reviews on a few products and yields sentiment of each review using a GRU neural network model trainned with 1467106 apperal reviews in Amazon.com. It also then dissembles each review by part of speech and displays the frequency of important words to learn what made customer feel negative as well as positive about the product. Since this app is only for a learning and experiment purpose, it scrapes and analyze only a few Amazon product pages real-time. **However, any further development is possible including a sentiment search and a real-time product sentiment analyzer and dashboard for all products in any online stores.** &nbsp;")

st.markdown("### Sample: Recent Clothing Review Sentiment in Amazon.com")

url = "https://www.amazon.com/Bsubseach-Sleeve-Blouses-Bathing-Swimwear/product-reviews/B087CNRPF9/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"

ran_df = get_reviews(url)
st.table(ran_df)
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
    st.table(w_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(w_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(w_df)

if option == "Men's Clothing":
    m_url = "https://www.amazon.com/Fruit-Loom-Eversoft-Cotton-T-Shirt/product-reviews/B0B2WV23RL/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    m_df = get_reviews(m_url)
    st.table(m_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(m_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(m_df)

if option == "Girls' Clothing":
    s_url = "https://www.amazon.com/The-Childrens-Place-Girls-Leggings/product-reviews/B09XWRT6RK/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    s_df = get_reviews(s_url)
    st.table(s_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(s_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(s_df)

if option == "Boys' Clothing":
    b_url = "https://www.amazon.com/Marvel-Avengers-Athletic-Shorts-Packs/product-reviews/B072WC6Y1G/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    b_df = get_reviews(b_url)
    st.table(b_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Positive Reviews : Strengths of the Product")
    positive_plot(b_df)
    st.markdown("&nbsp;")
    st.markdown("#### Finding Clues of Negative Reviews : Defects of the Product")
    negative_plot(b_df)

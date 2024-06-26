import re
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
stop_words_indonesian = set(stopwords.words('indonesian'))
stop_words_english = set(stopwords.words('english'))

def remove_emoticon_documents(df):
    emoticon_pattern = r'^[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF\s]+$'
    non_alpha_pattern = r'^[^a-zA-Z]*$'
    number_pattern = r'^\d+$'
    def is_unwanted(text):
        if re.match(emoticon_pattern, text):
            return True
        if re.match(non_alpha_pattern, text):
            return True
        if re.match(number_pattern, text):
            return True
        return False
    df['content'] = df['content'].apply(lambda x: '' if is_unwanted(str(x)) else x)
    df = df[df['content'].str.strip() != '']
    return df

def translate_text(text):
    translated_text = GoogleTranslator(source='en', target='id').translate(text)
    return translated_text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]
    
# Fungsi untuk menghapus emotikon
def remove_emoticons(text):
    emoticon_pattern = r'[\U00010000-\U0010ffff]'
    text = re.sub(emoticon_pattern, '', text)
    return text

# Fungsi untuk menghapus tanda baca dan angka
def remove_punctuation_and_numbers(text):
    punctuation_pattern = r'[^\w\s]'
    text = re.sub(punctuation_pattern, '', text)
    return text

def tambahkan_spasi_setelah_tanda_baca(teks):
    pola = re.compile(r'([.,!?/])(?![\s])')
    teks = pola.sub(r'\1 ', teks)
    return teks

def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def get_sentiment_label_and_polarity(text):
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores(text)
    compound_score = polarity_scores['compound']

    if compound_score > 0:
        sentiment_label = "positif"
    elif compound_score < 0:
        sentiment_label = "negatif"
    else:
        sentiment_label = "netral"

    return sentiment_label, compound_score

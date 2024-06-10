import re
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
stop_words_indonesian = set(stopwords.words('indonesian'))
stop_words_english = set(stopwords.words('english'))

def remove_emoticon_documents(df):
    emoticon_pattern = r'^[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]+$'
    df['content'] = df['content'].apply(lambda x: '' if re.match(emoticon_pattern, str(x)) else x)
    df = df[df['content'] != '']
    return df

def translate_text(text):
    translated_text = GoogleTranslator(source='en', target='id').translate(text)
    return translated_text

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words_indonesian]
    return ' '.join(filtered_tokens)

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

def lemmatize_text(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokenized_text]

def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def get_sentiment_label_and_polarity(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment_label = "positif"
    elif polarity < 0:
        sentiment_label = "negatif"
    else:
        sentiment_label = "netral"

    return sentiment_label, polarity

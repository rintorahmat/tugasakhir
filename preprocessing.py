from clean import remove_emoticon_documents, tambahkan_spasi_setelah_tanda_baca, remove_emoticons, remove_punctuation_and_numbers, lemmatize_text, stem_text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

def preprocess_data(data):
    try:
        data = remove_emoticon_documents(data)
        data['Spacing'] = data['content'].apply(tambahkan_spasi_setelah_tanda_baca)
        data['HapusEmoticon'] = data['Spacing'].apply(remove_emoticons)
        data['HapusTandaBaca'] = data['HapusEmoticon'].apply(remove_punctuation_and_numbers)
        data['LowerCasing'] = data['HapusTandaBaca'].str.lower()
        data['Tokenizing'] = data['LowerCasing'].apply(word_tokenize)
        data['Lemmatized'] = data['Tokenizing'].apply(lemmatize_text)
        data['Lemmatized'] = data['Lemmatized'].apply(lambda x: ' '.join(x))
        data['Lemmatized'] = data['Lemmatized'].astype(str)
        data['Stemmed'] = data['Lemmatized'].apply(stem_text)
        stop_words  = set(stopwords.words('indonesian'))
        stop_words.update(['dan','yang','saya','itu','untuk','ini','aja','jadi','lagi','bisa','di','nya','ada','yg','sih','ya','pa','tapi','ga','apk','kalo','gak'])
        data['StopWord'] = data['Stemmed'].str.replace('[^\w\s]','')
        data['StopWord'] = data['StopWord'].apply(lambda x:' '.join([word for word in x.split() if word not in stop_words]))

        return {"message": "sukses", "data": data[['content', 'Spacing', 'HapusEmoticon', 'HapusTandaBaca','LowerCasing', 'Tokenizing', 'Lemmatized', 'Stemmed', 'StopWord']].to_dict(orient='records')}
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return {"error": str(e)}

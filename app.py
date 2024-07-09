import io
import logging
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import os
import shutil
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from clean import remove_emoticon_documents, remove_emoticons, remove_punctuation_and_numbers, tambahkan_spasi_setelah_tanda_baca, translate_text, get_sentiment_label_and_polarity, stem_text, remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine, Column, Integer, String, BLOB, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from wordcloud import WordCloud
from pydantic import BaseModel

app = FastAPI()

split_data_storage = {}
results_storage = {}

DATABASE_SERVER_URL = "mysql+pymysql://admin:admin@34.30.136.207"
DATABASE_NAME = "tugasakhir"

temp_engine = create_engine(DATABASE_SERVER_URL)
with temp_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"))

DATABASE_URL = f"mysql+pymysql://admin:admin@34.30.136.207/{DATABASE_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FileModel(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre(Base):
    __tablename__ = "hasilpre"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre1(Base):
    __tablename__ = "hasildeleteline"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre2(Base):
    __tablename__ = "hasiltrans"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre3(Base):
    __tablename__ = "hasilspacing"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre4(Base):
    __tablename__ = "hasillowercasing"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre5(Base):
    __tablename__ = "hasildeleteemot"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre6(Base):
    __tablename__ = "hasilhapustandabaca"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre7(Base):
    __tablename__ = "hasiltoken"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre8(Base):
    __tablename__ = "hasilstopword"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)

class HasilPre9(Base):
    __tablename__ = "hasilstem"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    content = Column(BLOB)


Base.metadata.create_all(bind=engine)
@app.on_event("startup")
def on_startup():
    # Create all tables
    Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = "uploads"

def create_upload_dir():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

create_upload_dir()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SplitDataRequest(BaseModel):
    file_id: int
    test_size: float

class ClassificationResult(BaseModel):
    classification_report: dict
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

def save_preprocessed_data(data):
    file_path = os.path.join(UPLOAD_DIR, 'preprocessed_data.csv')
    data.to_csv(file_path, index=False)

def splitdata(file_id: int, test_size: float):
    logging.debug("Memulai proses file.")
    db = SessionLocal()
    db_file = db.query(HasilPre).filter(HasilPre.id == file_id).first()
    db.close()
    if db_file is None:
        raise HTTPException(status_code=404, detail="File data hasil Preprocessing tidak ditemukan")
    content = db_file.content
    print(f"Processing file: {db_file.filename}")
    data = pd.read_csv(io.BytesIO(content))
    if data.empty:
        raise HTTPException(status_code=400, detail="No data found in the file")

    X = data['Stemmed']
    y = data['SentimentLabel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logging.debug(f"Jumlah data dalam set pelatihan: {len(X_train)}")
    logging.debug(f"Jumlah data dalam set pengujian: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]
    
def map_score_to_sentiment(score):
            if score in [1, 2]:
                return 'negatif'
            elif score == 3:
                return 'netral'
            elif score in [4, 5]:
                return 'positif'
            else:
                return 'undefined'

@app.get("/")
async def read_root():
    return {"message": "Hello, world!"}

@app.post("/upload")
async def process(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.csv')):
            return {"error": "Only files with .csv extensions are allowed."}

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)

        with open(file_location, "rb") as file_object:
            file_content = file_object.read()

        print(file_content)
        db = SessionLocal()
        db_file = FileModel(
            filename=file.filename,
            content=file_content
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()
        print (db_file.filename)
        
        return {
            "info": f"File '{file.filename}' successfully uploaded.",
            "id": db_file.id,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/process/{file_id}")
async def process(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File not found")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="No data found in the file")
        if 'content' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'content' column found in the file")
        data = data[['content','score']]
        print(data)
        data['NilaiAktual'] = data['score'].map(map_score_to_sentiment)
        data = remove_emoticon_documents(data)
        data['Translated'] = data['content'].apply(translate_text)
        data['Space'] = data['Translated'].apply(tambahkan_spasi_setelah_tanda_baca)
        data['LowerCasing'] = data['Space'].str.lower()
        data['DeleteEmotikon'] = data['LowerCasing'].apply(remove_emoticons)
        data['HapusTandaBaca'] = data['DeleteEmotikon'].apply(remove_punctuation_and_numbers)
        data['Tokenizing'] = data['HapusTandaBaca'].apply(word_tokenize)
        data['Tokenizing'] = data['Tokenizing'].astype(str)
        data['StopWord'] = data['Tokenizing'].apply(lambda x: remove_stopwords(eval(x)))
        data['StopWord'] = data['StopWord'].astype(str)
        data['Stemmed'] = data['StopWord'].apply(stem_text)
        data[['SentimentLabel', 'Polarity']] = data['Stemmed'].apply(lambda x: pd.Series(get_sentiment_label_and_polarity(x)))

        sentiment_counts = data[['SentimentLabel']].value_counts()
        netral = int(sentiment_counts.get('netral', 0))
        positif  = int(sentiment_counts.get('positif', 0))
        negatif = int (sentiment_counts.get('negatif', 0))
        
        data['StopWord'] = data['StopWord'].astype(str)
        all_text = ''.join(data['StopWord'])

        wordcloud = WordCloud(width=1000, height=500, max_font_size=150, random_state=42).generate(all_text)
        buffer = BytesIO()
        plt.figure(figsize=(10,6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        new_data = data[['content', 'Translated', 'Space','LowerCasing', 'DeleteEmotikon', 'HapusTandaBaca', 'Tokenizing', 'StopWord', 'Stemmed',  'NilaiAktual', 'SentimentLabel', 'Polarity'  ]]
        save_preprocessed_data(new_data)
        
        file_location = os.path.join(UPLOAD_DIR, 'preprocessed_data.csv')
        with open(file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_file = HasilPre(
            filename='preprocessed_data.csv',
            content=file_content
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()
        print (db_file.filename)

        return {
            'data': new_data.to_dict(orient='records'),
            'id': db_file.id,
            'label_netral': netral,
            'label_positif':  positif,
            'label_negatif': negatif,
            'wordcloud_base64': img_str,
        }
    
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return {"error": str(e)}

@app.get("/procesblankdata/{file_id}")
async def procesblankdata(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data upload tidak ditemukan")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        
        initial_row_count = len(data)

        if data.empty:
            raise HTTPException(status_code=400, detail="File data upload tidak ditemukan")
        if 'content' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'content' column found in the file")
        
        data = data[['content','score']]
        data['NilaiAktual'] = data['score'].map(map_score_to_sentiment)
        
        data = remove_emoticon_documents(data)
        
        final_row_count = len(data)
        number_of_rows_removed = initial_row_count - final_row_count
        
        print(f"Number of rows removed: {number_of_rows_removed}")

        new_data = data[['content', 'NilaiAktual']]
        save_preprocessed_data(new_data)
        
        file_location = os.path.join(UPLOAD_DIR, 'preprocessed_data.csv')
        with open(file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_file = HasilPre1(
            filename='preprocessed_data.csv',
            content=file_content
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()
        print(db_file.filename)

        return JSONResponse(content={
            'data': new_data.to_dict(orient='records'),
            'id': db_file.id,
            'rows_removed': number_of_rows_removed 
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/translated/{file_id}")
async def translated(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre1).filter(HasilPre1.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Delete Blank Line tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Delete Blank Line tidak ditemukan")
        if 'content' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'content' column found in the file")
        
        data['Translated'] = data['content'].apply(translate_text)
        
        translated_data = data[['Translated', 'NilaiAktual']]
        print(data)
        translated_file_location = os.path.join(UPLOAD_DIR, 'translated_data.csv')
        translated_data.to_csv(translated_file_location, index=False)
        
        with open(translated_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_translated_file = HasilPre2(
            filename='translated_data.csv',
            content=file_content
        )
        db.add(db_translated_file)
        db.commit()
        db.refresh(db_translated_file)
        db.close()
        print(db_translated_file.filename)

        return JSONResponse(content={
            'data': translated_data.to_dict(orient='records'),
            'id': db_translated_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/spacing/{file_id}")
async def spacing(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre2).filter(HasilPre2.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Translated tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Translated tidak ditemukan")
        if 'Translated' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'Translated' column found in the file")
        
        data['Space'] = data['Translated'].apply(tambahkan_spasi_setelah_tanda_baca)
        
        spaced_data = data[['Space', 'NilaiAktual']]

        spaced_file_location = os.path.join(UPLOAD_DIR, 'spaced_data.csv')
        spaced_data.to_csv(spaced_file_location, index=False)
        
        with open(spaced_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_spaced_file = HasilPre3(
            filename='spaced_data.csv',
            content=file_content
        )
        db.add(db_spaced_file)
        db.commit()
        db.refresh(db_spaced_file)
        db.close()
        print(db_spaced_file.filename)

        return JSONResponse(content={
            'data': spaced_data.to_dict(orient='records'),
            'id': db_spaced_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/lowercasing/{file_id}")
async def lowercasing(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre3).filter(HasilPre3.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Space tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Space tidak ditemukan")
        if 'Space' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'Space' column found in the file")
        
        data['LowerCasing'] = data['Space'].str.lower()
        
        lowercasing_data = data[['LowerCasing', 'NilaiAktual']]

        lowercase_file_location = os.path.join(UPLOAD_DIR, 'Lowercasing.csv')
        lowercasing_data.to_csv(lowercase_file_location, index=False)
        
        with open(lowercase_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_lowercasing_file = HasilPre4(
            filename='Lowercasing.csv',
            content=file_content
        )
        db.add(db_lowercasing_file)
        db.commit()
        db.refresh(db_lowercasing_file)
        db.close()
        print(db_lowercasing_file.filename)

        return JSONResponse(content={
            'data': lowercasing_data.to_dict(orient='records'),
            'id': db_lowercasing_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/delemot/{file_id}")
async def delemot(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre4).filter(HasilPre4.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Lowercasing tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Lowercasing tidak ditemukan")
        if 'LowerCasing' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'Space' column found in the file")
        
        data['DeleteEmotikon'] = data['LowerCasing'].apply(remove_emoticons)
        
        delemot_data = data[['DeleteEmotikon', 'NilaiAktual']]

        delemot_file_location = os.path.join(UPLOAD_DIR, 'deleteemotikon_data.csv')
        delemot_data.to_csv(delemot_file_location, index=False)
        
        with open(delemot_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_delemot_file = HasilPre5(
            filename='deleteemotikon_data.csv',
            content=file_content
        )
        db.add(db_delemot_file)
        db.commit()
        db.refresh(db_delemot_file)
        db.close()
        print(db_delemot_file.filename)

        return JSONResponse(content={
            'data': delemot_data.to_dict(orient='records'),
            'id': db_delemot_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/hapustandabaca/{file_id}")
async def hapustandabaca(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre5).filter(HasilPre5.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Delete Emotikon tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Delete Emotikon tidak ditemukan")
        if 'DeleteEmotikon' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'DeleteEmotikon' column found in the file")
        
        data['HapusTandaBaca'] = data['DeleteEmotikon'].apply(remove_punctuation_and_numbers)
        
        hapustandabaca_data = data[['HapusTandaBaca', 'NilaiAktual']]

        hapustandabaca_file_location = os.path.join(UPLOAD_DIR, 'HapusTandaBaca.csv')
        hapustandabaca_data.to_csv(hapustandabaca_file_location, index=False)
        
        with open(hapustandabaca_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_hapustandabaca_file = HasilPre6(
            filename='HapusTandaBaca.csv',
            content=file_content
        )
        db.add(db_hapustandabaca_file)
        db.commit()
        db.refresh(db_hapustandabaca_file)
        db.close()
        print(db_hapustandabaca_file.filename)

        return JSONResponse(content={
            'data': hapustandabaca_data.to_dict(orient='records'),
            'id': db_hapustandabaca_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/tokenize/{file_id}")
async def tokenize(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre6).filter(HasilPre6.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Remove Punctuation tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Remove Punctuation tidak ditemukan")
        if 'HapusTandaBaca' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'LowerCasing' column found in the file")
        
        data['Tokenizing'] = data['HapusTandaBaca'].apply(word_tokenize)
        data['Tokenizing'] = data['Tokenizing'].astype(str)
        
        tokenize_data = data[['Tokenizing', 'NilaiAktual']]

        tokenize_file_location = os.path.join(UPLOAD_DIR, 'Tokenizing.csv')
        tokenize_data.to_csv(tokenize_file_location, index=False)
        
        with open(tokenize_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_tokenize_file = HasilPre7(
            filename='Tokenizing.csv',
            content=file_content
        )
        db.add(db_tokenize_file)
        db.commit()
        db.refresh(db_tokenize_file)
        db.close()
        print(db_tokenize_file.filename)

        return JSONResponse(content={
            'data': tokenize_data.to_dict(orient='records'),
            'id': db_tokenize_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/stopword/{file_id}")
async def stopword(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre7).filter(HasilPre7.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Tokenizing tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Tokenizing tidak ditemukan")
        if 'Tokenizing' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'Tokenizing' column found in the file")
        
        data['StopWord'] = data['Tokenizing'].apply(lambda x: remove_stopwords(eval(x)))
        data['StopWord'] = data['StopWord'].astype(str)
        stopword_data = data[['StopWord', 'NilaiAktual']]

        stopword_file_location = os.path.join(UPLOAD_DIR, 'stopword.csv')
        stopword_data.to_csv(stopword_file_location, index=False)
        
        with open(stopword_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_stopword_file = HasilPre8(
            filename='stopword.csv',
            content=file_content
        )
        db.add(db_stopword_file)
        db.commit()
        db.refresh(db_stopword_file)
        db.close()
        print(db_stopword_file.filename)

        return JSONResponse(content={
            'data': stopword_data.to_dict(orient='records'),
            'id': db_stopword_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/stemmed/{file_id}")
async def stemmed(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre8).filter(HasilPre8.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil StopWord tidak ditemukan")
        print(f"Processing file: {db_file.filename}")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil StopWord tidak ditemukan")
        if 'StopWord' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'StopWord' column found in the file")
        
        data['Stemmed'] = data['StopWord'].apply(stem_text)
        
        stem_data = data[['Stemmed', 'NilaiAktual']]

        stemm_file_location = os.path.join(UPLOAD_DIR, 'stemmed.csv')
        stem_data.to_csv(stemm_file_location, index=False)
        
        with open(stemm_file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_lemma_file = HasilPre9(
            filename='stemmed.csv',
            content=file_content
        )
        db.add(db_lemma_file)
        db.commit()
        db.refresh(db_lemma_file)
        db.close()
        print(db_lemma_file.filename)

        return JSONResponse(content={
            'data': stem_data.to_dict(orient='records'),
            'id': db_lemma_file.id,
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/sentimenanalis/{file_id}")
async def sentimenanalis(file_id: int):
    try:
        db = SessionLocal()
        db_file = db.query(HasilPre9).filter(HasilPre9.id == file_id).first()
        db.close()
        if db_file is None:
            raise HTTPException(status_code=404, detail="File data hasil Stemmed tidak ditemukan")
        content = db_file.content
        data = pd.read_csv(io.BytesIO(content))
        if data.empty:
            raise HTTPException(status_code=400, detail="File data hasil Stemmed tidak ditemukan")
        if 'Stemmed' not in data.columns:
            raise HTTPException(status_code=400, detail="No 'Stemmed' column found in the file")
        
        data[['SentimentLabel', 'Polarity']] = data['Stemmed'].apply(lambda x: pd.Series(get_sentiment_label_and_polarity(x)))

        sentiment_counts = data[['Stemmed', 'NilaiAktual', 'SentimentLabel', 'Polarity']].value_counts()
        netral = int(sentiment_counts.get('netral', 0))
        positif  = int(sentiment_counts.get('positif', 0))
        negatif = int (sentiment_counts.get('negatif', 0))
        
        data['Stemmed'] = data['Stemmed'].astype(str)
        all_text = ''.join(data['Stemmed'])

        wordcloud = WordCloud(width=1000, height=500, max_font_size=150, random_state=42).generate(all_text)
        buffer = BytesIO()
        plt.figure(figsize=(10,6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        new_data = data[['content', 'Translated', 'Space','LowerCasing', 'DeleteEmotikon', 'HapusTandaBaca', 'Tokenizing', 'StopWord', 'Stemmed', 'NilaiAktual', 'SentimentLabel', 'Polarity'  ]]
        save_preprocessed_data(new_data)
        
        file_location = os.path.join(UPLOAD_DIR, 'preprocessed_data.csv')
        with open(file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_file = HasilPre(
            filename='preprocessed_data.csv',
            content=file_content
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()
        print (db_file.filename)

        return {
            'data': new_data.to_dict(orient='records'),
            'id': db_file.id,
            'label_netral': netral,
            'label_positif':  positif,
            'label_negatif': negatif,
            'wordcloud_base64': img_str,
        }
    
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
        return {"error": str(e)}

@app.post("/splitdata/{file_id}")
async def split_data_endpoint(file_id: int, request: SplitDataRequest):
    try:
        X_train, X_test, y_train, y_test = splitdata(file_id, request.test_size)
        
        split_data_storage[file_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'test_size': request.test_size
        }

        train_data = pd.DataFrame({'Stemmed': X_train, 'SentimentLabel': y_train})
        test_data = pd.DataFrame({'Stemmed': X_test, 'SentimentLabel': y_test})

        return {
            "message": "Preprocessing completed",
            "file_id": file_id,
            "train_data": train_data.to_dict(orient='records'),
            "test_data": test_data.to_dict(orient='records')
        }
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/klasifikasi/")
async def klasifikasi(file_id: int = Query(...), test_size: float = Query(...)):
    try:
        logging.info(f"Received request with file_id={file_id} and test_size={test_size}")

        if file_id not in split_data_storage or split_data_storage[file_id]['test_size'] != test_size:
            logging.info(f"Splitting data for file_id={file_id} with test_size={test_size}")
            X_train, X_test, y_train, y_test = splitdata(file_id, test_size)
            split_data_storage[file_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'test_size': test_size
            }
        else:
            logging.info(f"Using cached split data for file_id={file_id}")
            data = split_data_storage[file_id]
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

        if len(set(y_train)) <= 1:
            raise ValueError("The number of classes has to be greater than one; got only one class in the training set")

        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        logging.info(f"y_test shape: {y_test.shape}")

        X_train = X_train.fillna("")
        X_test = X_test.fillna("")

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        logging.info("Training the SVM model")
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train_vec, y_train)

        logging.info("Predicting the test data")
        y_pred = svm_model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = report['macro avg']['precision']
        recall_macro = report['macro avg']['recall']
        f1_macro = report['macro avg']['f1-score']

        report_df = pd.DataFrame(report).transpose()

        report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']]
        report_df = report_df.round(2)

        report_str = report_df.to_string()

        results_storage[file_id] = {
            'classification_report': report_str,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }

        return {
            'classification_report': report,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_preprocessed/{file_id}")
async def download_preprocessed(file_id: int):
    db = SessionLocal()
    db_file = db.query(HasilPre).filter(HasilPre.id == file_id).first()
    db.close()
    if db_file is None:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(UPLOAD_DIR, db_file.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='application/octet-stream', filename=db_file.filename)

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
from clean import remove_emoticon_documents, remove_emoticons, remove_punctuation_and_numbers, tambahkan_spasi_setelah_tanda_baca, translate_text, get_sentiment_label_and_polarity, stem_text, lemmatize_text, remove_stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from wordcloud import WordCloud
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/", StaticFiles(directory="static", html=True), name="static")

split_data_storage = {}
results_storage = {}

DATABASE_URL = "sqlite:///./tugaskahir.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FileModel(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(BLOB)

class HasilPre(Base):
    __tablename__ = "hasilpre"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(BLOB)

Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.DEBUG)

# Path to upload directory
UPLOAD_DIR = "uploads"

# Create upload directory if it doesn't exist
def create_upload_dir():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

create_upload_dir()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
        raise HTTPException(status_code=404, detail="File not found")
    content = db_file.content
    data = pd.read_csv(io.BytesIO(content))
    if data.empty:
        raise HTTPException(status_code=400, detail="No data found in the file")

    X = data['StopWord']  # Fitur
    y = data['Sentiment_Label']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logging.debug(f"Jumlah data dalam set pelatihan: {len(X_train)}")
    logging.debug(f"Jumlah data dalam set pengujian: {len(X_test)}")

    return X_train, X_test, y_train, y_test

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.get("/")
async def read_root():
    return {"message": "Hello, world!"}

@app.post("/upload")
async def process(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only files with .csv extensions are allowed.")

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)

        with open(file_location, "rb") as file_object:
            file_content = file_object.read()

        db = SessionLocal()
        db_file = FileModel(
            filename=file.filename,
            content=file_content
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        db.close()
        
        return {
            "info": f"File '{file.filename}' successfully uploaded.",
            "id": db_file.id,
        }
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/process/{file_id}")
async def process(file_id: int):
    try:
        logging.debug("Memulai proses file.")
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
        data = data[['content']]
        data = remove_emoticon_documents(data)
        jumlah_data_sesudah = len(data)
        print(f"Jumlah data setelah menghapus emotikon: {jumlah_data_sesudah}")
        print(data)
        data['Translated'] = data['content'].apply(translate_text)
        data['Spacing'] = data['Translated'].apply(tambahkan_spasi_setelah_tanda_baca)
        data['HapusEmoticon'] = data['Spacing'].apply(remove_emoticons)
        data['HapusTandaBaca'] = data['HapusEmoticon'].apply(remove_punctuation_and_numbers)
        data['LowerCasing'] = data['HapusTandaBaca'].str.lower()
        data['Tokenizing'] = data['LowerCasing'].apply(word_tokenize)
        data['Lemmatized'] = data['Tokenizing'].apply(lemmatize_text)
        data['Lemmatized'] = data['Lemmatized'].apply(lambda x: ' '.join(x))
        data['Stemmed'] = data['Lemmatized'].apply(stem_text)
        data['StopWord'] = data['Stemmed'].apply(remove_stopwords)
        data[['Sentiment_Label', 'Polarity']] = data['StopWord'].apply(lambda x: pd.Series(get_sentiment_label_and_polarity(x)))

        sentiment_counts = data[['Sentiment_Label']].value_counts()
        netral = int(sentiment_counts.get('netral', 0))
        positif  = int(sentiment_counts.get('positif', 0))
        negatif = int (sentiment_counts.get('negatif', 0))
        
        data['StopWord'] = data['StopWord'].astype(str)
        all_text = ''.join(data['StopWord'])

        wordcloud = WordCloud(width=1000, height=500, max_font_size=150, random_state=42).generate(all_text)
        # Konversi gambar wordcloud ke base64
        buffer = BytesIO()
        plt.figure(figsize=(10,6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        new_data = data[['content', 'Spacing', 'HapusEmoticon', 'HapusTandaBaca','LowerCasing', 'Tokenizing', 'Lemmatized', 'StopWord', 'Sentiment_Label', 'Polarity'  ]]
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
        # Split the data
        X_train, X_test, y_train, y_test = splitdata(file_id, request.test_size)
        
        # Store split data in temporary storage
        split_data_storage[file_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'test_size': request.test_size  # Store test_size for validation
        }

        # Convert split data to DataFrames
        train_data = pd.DataFrame({'StopWord': X_train, 'Sentiment_Label': y_train})
        test_data = pd.DataFrame({'StopWord': X_test, 'Sentiment_Label': y_test})

        # Return success message along with train and test data
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

        # Check if data is already in storage
        if file_id not in split_data_storage or split_data_storage[file_id]['test_size'] != test_size:
            logging.info(f"Splitting data for file_id={file_id} with test_size={test_size}")
            X_train, X_test, y_train, y_test = splitdata(file_id, test_size)
            split_data_storage[file_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'test_size': test_size  # Store test_size for validation
            }
        else:
            logging.info(f"Using cached split data for file_id={file_id}")
            data = split_data_storage[file_id]
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']

        # Validate that y_train has more than one class
        if len(set(y_train)) <= 1:
            raise ValueError("The number of classes has to be greater than one; got only one class in the training set")

        # Log shapes of the datasets
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

        # Convert the classification report to DataFrame
        report_df = pd.DataFrame(report).transpose()

        # Convert metrics to percentages
        report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']]
        report_df = report_df.round(2)

        # Convert DataFrame back to string for display
        report_str = report_df.to_string()

        # Save results in temporary storage
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
    return FileResponse(file_path, media_type='application/octet-stream', filename=db_file.filename)

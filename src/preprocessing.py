import json
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
# nltk.download('punkt_tab')
# # Ensure the resources are downloaded
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.data.path.append(r'C:/Users/joyri/AppData/Roaming/nltk_data/tokenizers')
#C:\Users\joyri\anaconda3\Lib\site-packages\nltk\tokenize\punkt.py
def load_file(folder_path, filename, filetype):
    """
    Reads a JSON file from a specified folder 

    Parameters:
        folder_path (str): The relative or absolute path to the folder containing the JSON file.
        filename (str): The name of the JSON file (including .json extension).

    Returns:
        pd.DataFrame: DataFrame containing the JSON data.
    """
    file_path = os.path.join(folder_path, filename)
        # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}' not found in folder '{folder_path}'")
    
    if filetype == 'json':
        with open(file_path,"r", encoding='utf-8') as file:
            data = json.load(file)
    elif filetype == 'csv':
        data = pd.read_csv(file_path)
    elif filetype == 'pkl':
        data = joblib.load(file_path)
    else:
        print('Invalid file type')

    return data

def create_dataframe_from_json(data) ->pd.DataFrame:  
    '''
    This function is used to create dataframe from the JSON that is loaded
    '''
    records = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    records.append({
                        "Title": article['title'],
                        "Context": context,
                        "Question": qa['question'],
                        "Question_ID": qa['id'],
                        "Answer_Text": answer['text'],
                        "Answer_Start": answer['answer_start'],
                        "Is_Impossible": qa['is_impossible']
                    })
    df = pd.DataFrame(records)
    return df

def train_vectorizer(text, vector_file_path):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),stop_words='english')
    text_vector = vectorizer.fit_transform(text)
    joblib.dump(vectorizer, vector_file_path) #"tfidf_vectorizer.pkl" save
    return text_vector

def use_vectorizer(text,vectorizer):
    # Convert titles to TF-IDF vectors
    text_vector = vectorizer.transform(text)
    return text_vector

def save_processed_dataframe(df_class, process_data_folder_path, process_data_file_name):
    file_path = os.path.join(process_data_folder_path,process_data_file_name)
    df_class.to_csv(file_path)

def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN

class LemmaTokenizer:
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    tokens = word_tokenize(doc) # word_tokenize is a function in NLTK
    words_and_tags = nltk.pos_tag(tokens)
    return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) \
            for word, tag in words_and_tags]

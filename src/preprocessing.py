import json
import os

import joblib
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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
        raise FileNotFoundError(
            f"File '{filename}' not found in folder '{folder_path}'"
        )

    if filetype == "json":
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif filetype == "csv":
        data = pd.read_csv(file_path)
    elif filetype == "pkl":
        data = joblib.load(file_path)
    else:
        print("Invalid file type")

    return data


def create_dataframe_from_json(data) -> pd.DataFrame:
    """
    This function is used to create dataframe from the JSON that is loaded
    """
    records = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                for answer in qa["answers"]:
                    records.append(
                        {
                            "Title": article["title"],
                            "Context": context,
                            "Question": qa["question"],
                            "Question_ID": qa["id"],
                            "Answer_Text": answer["text"],
                            "Answer_Start": answer["answer_start"],
                            "Is_Impossible": qa["is_impossible"],
                        }
                    )
    df = pd.DataFrame(records)
    return df


def train_vectorizer(text, vector_file_path):
    """
    Trains a TF-IDF vectorizer on the given text data and saves the trained model.

    Parameters:
    text (list of str): A list of text documents to be vectorized.
    vector_file_path (str): The file path to save the trained TF-IDF vectorizer.

    Returns:
    sparse matrix: The transformed text as a sparse matrix of TF-IDF features.

    Saves:
    A trained TF-IDF vectorizer as a .pkl file for later use.
    """

    print("Starting TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words="english")
    text_vector = vectorizer.fit_transform(text)
    print("Vectorization complete!")

    joblib.dump(vectorizer, vector_file_path)  # "tfidf_vectorizer.pkl" save
    return text_vector


def use_vectorizer(text, vectorizer):
    """
    Transforms input text using a pre-trained TF-IDF vectorizer.

    Parameters:
    text (list of str): A list of text documents to be transformed.
    vectorizer (TfidfVectorizer): A trained TF-IDF vectorizer.

    Returns:
    sparse matrix: The transformed text as a sparse matrix of TF-IDF features.
    """
    text_vector = vectorizer.transform(text)
    return text_vector


def save_processed_dataframe(
    df_class, process_data_folder_path, process_data_file_name
):
    """
    Saves a processed DataFrame as a CSV file.

    Parameters:
    df_class (pandas.DataFrame): The DataFrame to be saved.
    process_data_folder_path (str): The directory where the CSV file should be stored.
    process_data_file_name (str): The name of the CSV file.

    Saves:
    A CSV file containing the processed data.
    """
    file_path = os.path.join(process_data_folder_path, process_data_file_name)
    df_class.to_csv(file_path)


def get_wordnet_pos(treebank_tag):
    """
    Maps NLTK part-of-speech tags to WordNet POS tags.

    Parameters:
    treebank_tag (str): A POS tag from NLTK's pos_tag function.

    Returns:
    str: The corresponding WordNet POS tag.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class LemmaTokenizer:
    """
    A custom tokenizer that tokenizes and lemmatizes input text using NLTK.

    Methods:
    __call__(doc): Tokenizes and lemmatizes a given text document.
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenizes and lemmatizes the input document.

        Parameters:
        doc (str): A text document to be tokenized and lemmatized.

        Returns:
        list of str: A list of lemmatized words.
        """
        tokens = word_tokenize(doc)  # word_tokenize is a function in NLTK
        words_and_tags = nltk.pos_tag(tokens)
        return [
            self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))
            for word, tag in words_and_tags
        ]

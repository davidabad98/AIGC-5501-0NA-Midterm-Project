import json
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

def vectorize_text(text,vectorizer, single_text: bool):
    # Convert titles to TF-IDF vectors

    if single_text:
        text_vector = vectorizer.transform([text])
    else:  
        vectorizer = TfidfVectorizer(stop_words='english')    
        text_vector = vectorizer.fit_transform(text)
    return text_vector

def create_cluster(df, number_of_cluster = 10) ->pd.DataFrame:
    '''
    We are going to create clusters based on the context of each row
    and then assign a name to each cluster
    '''

    input_text = vectorize_text(df['Context'],None, False)
    # Apply K-Means clustering
    num_clusters = number_of_cluster  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(input_text)
    # Determine the most common cluster for each title
    most_common_clusters = df.groupby('Title')['cluster'].agg(lambda x: x.value_counts().idxmax())

    # Map the most frequent cluster back to the dataframe
    df['cluster'] = df['Title'].map(most_common_clusters)

    # This is the dictionary used to map the cluster names to cluster
    name_dict = {0:'Education',1:'Politics',2:'Football',3:'Countries and Empires', 4:'Warfare', 6:'Cities', 7:'Science', 8:'Language', 9:'Music'}

    # Removing cluster 5 which is a mix
    df_class = df.loc[df['cluster']!=5,:]
    # Mapping the name to each cluster
    df_class['topic'] = df_class['cluster'].map(name_dict)

    return df_class

def save_processed_dataframe(df_class, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    df_class.to_csv(file_path)

def save_model_and_metric(model,vectorizer,accuracy,report,folder_path, model_filename,vector_filename, metric_filename):
        # Save Model and Vectorizer
    model_file_path = os.path.join(folder_path, model_filename)
    vector_file_path = os.path.join(folder_path, vector_filename)
    joblib.dump(model, model_file_path)
    joblib.dump(vectorizer, vector_file_path) #"tfidf_vectorizer.pkl"
    # Save Model Metrics to a Text File
    metric_file_path = os.path.join(folder_path, metric_filename)
    with open(metric_file_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Model and metrics saved successfully.")

if __name__ == '__main__':
    '''
    If you run this file it will create a csv file from the json, that has been clustered into
    processed folder
    '''
    
    folder_path = "./dataset/raw"  # Adjust path as needed
    filename = "wiki.json"
    data = load_file(folder_path, filename, 'json')
    df = create_dataframe_from_json(data)
    df_class = create_cluster(df, number_of_cluster = 10)
    save_folder_path = "./dataset/processed"
    save_filename = "clustered_data.csv"
    save_processed_dataframe(df_class, save_folder_path, save_filename)
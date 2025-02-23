from preprocessing import *
from model import *

def create_data_model():
    source_folder_path = "./dataset/raw"  # Adjust path as needed
    source_filename = "wiki.json"
    save_folder_path = "./dataset/processed"
    save_filename = "clustered_data.csv"
    # Load data
    data = load_file(source_folder_path, source_filename, 'json')
    # Create dataframe
    df = create_dataframe_from_json(data)
    # Clustering job
    df_class = create_cluster(df, number_of_cluster = 10)
    # Save csv file
    save_processed_dataframe(df_class, save_folder_path, save_filename)

    # load csv
    # classification job
    # save the models 
    cm = classificationModel()
    cm.question_classification_model()

if __name__ == '__main__':
    create_data_model()
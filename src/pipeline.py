from preprocessing import *
from classification import *
from clustering import *
import warnings
warnings.filterwarnings("ignore")

def create_data_model(choose_model='Logistic'):
    source_folder_path = "./dataset/raw"  # Adjust path as needed
    source_filename = "wiki.json"
    process_data_folder_path = "./dataset/processed"
    process_data_file_name = "clustered_data.csv"
    # Load data
    data = load_file(source_folder_path, source_filename, 'json')
    #Create dataframe
    df = create_dataframe_from_json(data)
    # Clustering job
    cl = clusteringModel()
    df_class = cl.create_cluster(df)
    # Save csv file
    save_processed_dataframe(df_class, process_data_folder_path, process_data_file_name)

    # load csv, classification job, save the models 
    cm = classificationModel()
    #cm.question_classification_model()
    if choose_model == 'Logistic':
        cm.question_classification_model()
    else:
        cm.question_NB_classification_model()
if __name__ == '__main__':
    create_data_model(choose_model='Logistic')
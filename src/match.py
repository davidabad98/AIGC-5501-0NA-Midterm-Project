from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import *
from classification import *

def calculate_cos_similarity(tfidf_matrix):
    # Compute cosine similarity between user question and all dataset questions
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    return cosine_similarities

# Function to find best matching question
def find_best_match(user_question, df, predicted_cluster):
    # Filter DataFrame to include only rows where final_cluster matches selected_cluster
    filtered_df = df[df["cluster"] == predicted_cluster]

     # converting to list
    data_text = filtered_df["Question"].tolist()  # Extract question from the relevant class
    text = data_text+[user_question]
    cm = classificationModel()
    model_folder_path = cm.model_folder_path
    vector_filename = cm.vector_filename
    vectorizer = load_file(model_folder_path,vector_filename, 'pkl')
    tfidf_matrix = use_vectorizer(text,vectorizer)
    cosine_similarities = calculate_cos_similarity(tfidf_matrix)

    best_match_index = cosine_similarities.argmax()  # Get index of highest similarity
    best_match_score = cosine_similarities[best_match_index] * 100  # Convert to percentage
    
    # Retrieve best matching question and corresponding answer
    best_question = filtered_df.iloc[best_match_index]["Question"]
    best_answer = filtered_df.iloc[best_match_index]["Answer_Text"]

    return best_question, best_answer, round(best_match_score, 2)
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

    # best_match_index = cosine_similarities.argmax()  # Get index of highest similarity
    # best_match_score = cosine_similarities[best_match_index] * 100  # Convert to percentage
    
    # # Retrieve best matching question and corresponding answer
    # best_question = filtered_df.iloc[best_match_index]["Question"]
    # best_answer = filtered_df.iloc[best_match_index]["Answer_Text"]

    # return best_question, best_answer, round(best_match_score, 2)

    user_similarity_scores = cosine_similarities[:-1]  # Exclude self-comparison
        # Get top 3 matches
    top_3_indices = user_similarity_scores.argsort()[-3:][::-1]  # Sort and get top 3
    top_3_scores = user_similarity_scores[top_3_indices] * 100  # Convert to percentage

        # Retrieve the best match
    best_match_index = top_3_indices[0]
    best_question = filtered_df.iloc[best_match_index]["Question"]
    best_answer = filtered_df.iloc[best_match_index]["Answer_Text"]
    best_context = filtered_df.iloc[best_match_index]["Context"]
    best_match_score = round(top_3_scores[0], 2)

    top_3_questions = []
    for idx, score in zip(top_3_indices, top_3_scores):
        top_3_questions.append({
            "question": filtered_df.iloc[idx]["Question"],
            "context": filtered_df.iloc[idx]["Context"],
            "answer": filtered_df.iloc[idx]["Answer_Text"],
            "similarity_score": round(score, 2)
        })

    return {
        "best_match": {
            "question": best_question,
            "context": best_context,
            "answer": best_answer,
            "similarity_score": best_match_score,
        },
        "top_3_similar": top_3_questions,
    }
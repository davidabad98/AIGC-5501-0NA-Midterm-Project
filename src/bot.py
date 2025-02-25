from preprocessing import *
from match import *
from classification import *
import warnings
warnings.filterwarnings("ignore")
# Example usage
import time

def run_bot():
    # The while loop will keep asking for prompts
    while True:
        user_question =  input("User Question: ")
        if user_question == 'qt':
            break
        
        start = time.time()
        # Calling the classification model
        cm = classificationModel()
        # Using the model to fetch the class of user question
        predicted_cluster = cm.classify_question(user_question)
        
        # Load the processed data
        df = load_file(cm.folder_path, cm.filename, 'csv')
        # This function finds the best match of user question ffrom question dataset
        result  = find_best_match(user_question, df, predicted_cluster)
        end = time.time()

        # Parse "results" to get the best match questions and answers as well as contexts
        best_match = result["best_match"]
        top_3_similar = result["top_3_similar"]
        # If the score is more that 80 we give the answer
        if best_match["similarity_score"] > 80:
            print(f"\nBest Matched Question: {best_match['question']}")
            print(f"Match Score: {best_match['similarity_score']}%")
            print(f"Answer: {best_match['answer']}")
        # Otherwise it will print the context and show related questions
        else:
            print("\nWe do not have the exact answer to your question, but here is some related details: ")
            print(f"\nContext: {best_match['context']}")
            print("\nAre you interested to know about: ")
            for i, item in enumerate(top_3_similar, 1):
                print(f"\nSimilar Question: {item['question']}")
                print(f"Similarity Score: {item['similarity_score']}%")
        
        print(f"\nProcessing Time: {end - start:.2f} seconds.\n")
if __name__ == '__main__':
    run_bot()
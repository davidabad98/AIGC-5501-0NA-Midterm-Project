from preprocessing import *
from match import *
from classification import *
import warnings
warnings.filterwarnings("ignore")
# Example usage
import time

def run_bot():
    while True:
        user_question =  input("User Question: ")
        if user_question == 'qt':
            break

        start = time.time()
        cm = classificationModel()
        predicted_cluster = cm.classify_question(user_question)
        
        df = load_file(cm.folder_path, cm.filename, 'csv')
        result  = find_best_match(user_question, df, predicted_cluster)
        end = time.time()

        best_match = result["best_match"]
        top_3_similar = result["top_3_similar"]

        if best_match["similarity_score"] > 80:
            print(f"\nBest Matched Question: {best_match['question']}")
            print(f"Match Score: {best_match['similarity_score']}%")
            print(f"Answer: {best_match['answer']}")
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
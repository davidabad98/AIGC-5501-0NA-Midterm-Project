from preprocessing import *
from match import *
from classification import *
import warnings
warnings.filterwarnings("ignore")
# Example usage

def run_bot():
    user_question =  "When was beyonce born?"
    cm = classificationModel()
    predicted_cluster = cm.classify_question(user_question)
    
    df = load_file(cm.folder_path, cm.filename, 'csv')
    best_question, best_answer, match_score = find_best_match(user_question, df, predicted_cluster)

    # Print result
    print(f"User Question: {user_question}")
    print(f"Best Matched Question: {best_question}")
    print(f"Match Score: {match_score}%")
    print(f"Answer: {best_answer}")

if __name__ == '__main__':
    run_bot()
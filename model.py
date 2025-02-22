from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import *

'''
We create a classification model based on the question to predic the cluster that the question belong to.
This model is meant to reduce the amount of text that needs to be matched when the user asks a 
question.
'''

def question_classification_model():

    folder_path = './dataset/processed'
    filename = 'clustered_data.csv'
    df_class = load_file(folder_path, filename, 'csv')
    # Load Data (Assuming df contains 'title' and 'final_cluster')
    X = df_class['Question']  # Text data
    y = df_class['cluster']  # Target labels

    # Convert Text to Numerical Features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    #  Split Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    #  Make Predictions
    y_pred = model.predict(X_test)

        # Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy is = {}".format(accuracy))
    print(report)
    # save the model and metric
    model_folder_path = './model'
    model_filename = "logistic_regression_model.pkl"
    metric_filename = "metrics.txt"
    save_model_and_metric(model,accuracy,report,model_folder_path, model_filename, metric_filename)


if __name__ =='__main__':
    question_classification_model()
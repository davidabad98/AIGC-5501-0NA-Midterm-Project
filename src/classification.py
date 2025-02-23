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
class classificationModel:

    def __init__(self):
        self.folder_path = './dataset/processed'
        self.filename = 'clustered_data.csv'
        self.model_folder_path = './model'
        self.model_filename = "logistic_regression_model.pkl"
        self.vector_filename = "classify_vectorizer.pkl"
        self.metric_filename = "metrics.txt"

    def question_classification_model(self):   

        df_class = load_file(self.folder_path, self.filename, 'csv')
        # Load Data (Assuming df contains 'title' and 'final_cluster')
        X = df_class['Question']  # Text data
        y = df_class['cluster']  # Target labels

        # # Convert Text to Numerical Features using TF-IDF
        # vectorizer = TfidfVectorizer(stop_words = 'english')
        # X_tfidf = vectorizer.fit_transform(X)

        vector_file_path = os.path.join(self.model_folder_path, self.vector_filename)
        X_tfidf = train_vectorizer(X, vector_file_path)

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
        self.save_model_and_metric(model,accuracy,report)

    def classify_question(self,question):        
        model = load_file(self.model_folder_path, self.model_filename, 'pkl')
        vectorizer = load_file(self.model_folder_path,self.vector_filename, 'pkl')
        #question_tfidf = vectorize_text(question,vectorizer, True)
        question_tfidf = use_vectorizer([question],vectorizer)
        predicted_cluster = model.predict(question_tfidf)[0]  # Predict cluster
        print('Predicted cluster - {}'.format(predicted_cluster))
        return predicted_cluster
    
    def save_model_and_metric(self, model,accuracy,report):
        # Save Model
        model_file_path = os.path.join(self.model_folder_path, self.model_filename)
        joblib.dump(model, model_file_path)
        
        # Save Model Metrics to a Text File
        metric_file_path = os.path.join(self.model_folder_path, self.metric_filename)
        with open(metric_file_path, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write("Classification Report:\n")
            f.write(report)

        print("Model and metrics saved successfully.")


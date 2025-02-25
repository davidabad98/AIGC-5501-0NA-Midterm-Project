import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocessing import *

"""
We create a classification model based on the question to predic the cluster that the question belong to.
This model is meant to reduce the amount of text that needs to be matched when the user asks a 
question.
"""


class classificationModel:

    def __init__(self):
        self.folder_path = "./dataset/processed"
        self.filename = "clustered_data.csv"
        self.model_folder_path = "./model"
        self.model_filename = "classification_model.pkl"
        self.vector_filename = "classify_vectorizer.pkl"
        self.metric_filename = "metrics.txt"
        self.model_naive_filename = "naive_bayes_model.pkl"

    # This function creates the logistic regression model
    def question_classification_model(self):

        df_class = load_file(self.folder_path, self.filename, "csv")
        # Load Data (Assuming df contains 'title' and 'final_cluster')
        X = df_class["Question"]  # Text data
        y = df_class["cluster"]  # Target labels

        vector_file_path = os.path.join(self.model_folder_path, self.vector_filename)
        X_tfidf = train_vectorizer(X, vector_file_path)

        #  Split Data into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )

        # Train Logistic Regression Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        #  Make Predictions
        y_pred = model.predict(X_test)

        # Evaluate Model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        num_clusters = 10
        cluster_labels = [
            str(i) for i in range(num_clusters)
        ]  # Labels: ["0", "1", ..., "9"]
        self.plot_confusion_matrix(y_test, y_pred, labels=cluster_labels)

        print("Accuracy is = {}".format(accuracy))
        print(report)
        # save the model and metric
        self.save_model_and_metric(model, accuracy, report)

    # This model creates the Naive Bayes model
    def question_NB_classification_model(self):
        # Load Data (Assuming df contains 'Question' and 'cluster')
        df_class = load_file(self.folder_path, self.filename, "csv")
        X = df_class["Question"]  # Text data
        y = df_class["cluster"]  # Target labels

        # Convert Text to Numerical Features using TF-IDF
        vector_file_path = os.path.join(self.model_folder_path, self.vector_filename)
        X_tfidf = train_vectorizer(X, vector_file_path)  # Your TF-IDF vectorizer logic

        # Split Data into Train and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )

        # Train Naive Bayes Model
        modelNB = MultinomialNB()
        modelNB.fit(X_train, y_train)

        # Make Predictions
        y_pred = modelNB.predict(X_test)

        # Evaluate Model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Accuracy is = {}".format(accuracy))
        print(report)

        # Save the model and metric
        self.save_model_and_metric(modelNB, accuracy, report)

    # This function takes the user question and uses the classification model to predict the class 
    def classify_question(self, question):
        # We are loading the treained model
        model = load_file(self.model_folder_path, self.model_filename, "pkl")
        # We are loading the trained vectorizer
        vectorizer = load_file(self.model_folder_path, self.vector_filename, "pkl")
        # Using the trained vectorizer on user query
        question_tfidf = use_vectorizer([question], vectorizer)
        # Predict the class of user question
        predicted_cluster = model.predict(question_tfidf)[0]  # Predict cluster
        print("Predicted cluster - {}".format(predicted_cluster))
        return predicted_cluster

    def save_model_and_metric(self, model, accuracy, report):
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

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        labels=None,
        title="Confusion Matrix",
        save_path="results/confusion_matrix.png",
    ):
        """
        Plots the confusion matrix for a classification model.

        Parameters:
        - y_true: Actual labels
        - y_pred: Predicted labels
        - labels: List of class labels (optional)
        - title: Title of the plot (default: "Confusion Matrix")
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)

        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

        print(f"Confusion matrix saved at: {save_path}")

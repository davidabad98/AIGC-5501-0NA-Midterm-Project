import os

from preprocessing import *


class clusteringModel:

    def __init__(self):
        self.cluster_vectorizer_folder_path = "./model"
        self.cluster_vectorizer_file_name = "cluster_vectorizer.pkl"
        self.number_of_cluster = 10

    def create_cluster(self, df) -> pd.DataFrame:
        """
        We are going to create clusters based on the context of each row
        and then assign a name to each cluster
        """
        # filepath to save vectorizer
        vector_file_path = os.path.join(
            self.cluster_vectorizer_folder_path, self.cluster_vectorizer_file_name
        )
        # Training a vectorizer with context and saving to file path
        input_text = train_vectorizer(df["Context"], vector_file_path)

        # Apply K-Means clustering
        num_clusters = self.number_of_cluster
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(input_text)

        # Determine the most common cluster for each title
        most_common_clusters = df.groupby("Title")["cluster"].agg(
            lambda x: x.value_counts().idxmax()
        )

        # Map the most frequent cluster back to the dataframe
        df["cluster"] = df["Title"].map(most_common_clusters)

        # This is the dictionary used to map the cluster names to cluster
        # name_dict = {0:'Politics',1:'History and Religion',2:'Cities',3:'Language', 4:'Warfare', 6:'Football', 7:'Countries and Empires', 8:'Universities', 9:'Music'}
        # topics = ['Music', 'Miscellenous', 'Politics', 'City', 'University', 'History and Religion', 'Football', 'War', 'Language']
        # Removing cluster 5 which is a mix
        # df_class = df.loc[df['cluster']!=5,:]
        # # Mapping the name to each cluster
        # df_class['topic'] = df_class['cluster'].map(name_dict)

        return df

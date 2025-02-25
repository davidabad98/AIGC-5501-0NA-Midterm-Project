# AIGC-5501-0NA-Midterm-Project

# Q&A Bot with Traditional Machine Learning

## Project Overview

This project builds a Question Answering (QA) system using traditional machine learning algorithms on the SQuAD dataset. The system predicts answers to user queries without using neural networks by leveraging TF-IDF vectorization, clustering (KMeans), and Logistic Regression.

## Project Workflow

### **Training Phase**

1. **Input Data**: Load the SQuAD dataset with `Context`, `Question`, and `Answer` columns.
2. **Vectorization**: Convert context text into numerical vectors using **TF-IDF vectorization**.
3. **Clustering**: Apply **K-Means** to cluster the context vectors.
4. **Store Clusters**: Save cluster labels in the dataset.
5. **Question Vectorization**: Convert questions into vectors using **TF-IDF**.
6. **Train Classifier**: Train a **Logistic Regression** model to predict cluster labels based on question vectors.
7. **Save Models**: Store the trained vectorizer, cluster model, and classifier for later use.

### **Testing Phase**

1. **User Input**: Accept a new question from the user.
2. **Cluster Prediction**: Use the **Logistic Regression model** to predict the relevant cluster.
3. **Filter Data**: Retrieve only the data belonging to the predicted cluster.
4. **Similarity Matching**: Use **TF-IDF and Cosine Similarity** to find the most relevant question.
5. **Answer Selection**:
   - If **similarity ≥ 80%** → Return the matched question and its answer.
   - If **similarity < 80%** → Show related context and similar questions, explaining that no exact match was found.

## Repository Structure
```
kmnist-optimizer-comparison/
│── dataset/               # Stores datasets (raw and processed)
│── src/                   # Source code for model training & evaluation
│   │── __init__.py        # Marks src as a package
│   │── clustering.py      # Implements KMeans clustering
│   │── preprocessing.py   # Implements data preprocessing
│   │── match.py           # Find best question match
│   │── download.py        # Download nltk dependencies
│   │── evaluation.py      # Has all Evaluation functions
│   │── pipeline.py        # Starts model training
│   │── bot.py             # Main script to run everything
│── results/               # Output files (graphs, logs, etc.)
│── README.md              # Project documentation
│── requirements.txt       # Dependencies for easy setup
│── .gitignore             # Ignore unnecessary files
```
## Getting Started
### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/davidabad98/AIGC-5501-0NA-Midterm-Project.git
   cd AIGC-5501-0NA-Midterm-Project
   ```
2. For Training Run:
   ```bash
   python src/pipeline.py
   ```
2. For Q&A Run:
   ```bash
   python src/bot.py
   ```

## Contributors
- **[David Abad](https://github.com/davidabad98)**
- **[Rizvan Nahif](https://github.com/joyrizvan)**
- **[Darshil Shah](https://github.com/darshil0811)**
- **[Navpreet Kaur Dusanje](https://github.com/Navpreet-Kaur-Dusanje)**



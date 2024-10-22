import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TRAIN_DATE_PATH = 'data/q_a.csv'
NUM_CLUSTERS = 15

class IntentRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.num_clusters = NUM_CLUSTERS
        self.df = None
        self.load_data(TRAIN_DATE_PATH)
        self.fit()

    def load_data(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def get_embedding(self, sentence):
        # Get embeddings for sentences
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():  # Prevent gradient calculation
            outputs = self.model(**inputs)
        res = np.array(outputs.last_hidden_state.mean(dim=1).detach().tolist())  # Return as numpy array
        return res

    def fit(self):
        # Apply embedding on all questions
        self.df['embedding'] = self.df['question'].apply(self.get_embedding)

        # Stack embeddings into a numpy array for clustering
        X = np.vstack(self.df['embedding'].values)

        # Apply KMeans Clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        self.df['cluster'] = kmeans.fit_predict(X)

        # Cluster labels, manually generate it for now, please reference ipynb
        # TODO: Can integrate openai or azure to summarize the question then generate labels
        cluster_labels = {
            0: "Course inquiries",
            1: "Chatbot itself",
            2: "Waitlisted inquiries",
            3: "Problem-solving",
            4: "Class recommendations",
            5: "First day",
            6: "Financial aid",
            7: "Parking",
            8: "Graduation",
            9: "Easy class",
            10: "First day of classes",
            11: "Rumibot creator",
            12: "Scholarship",
            13: "Course petition submission",
            14: "Math problem"
        }

        self.df['intent'] = self.df['cluster'].map(cluster_labels)

    def calculate_similarity(self, embedding1, embedding2):
        # Calculate cosine similarity between two embeddings
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def classify_intent(self, user_input):
        # Generate embedding for the new user input
        user_embedding = self.get_embedding(user_input)

        max_similarity = -1
        detected_intent = None

        # Loop through each intent and calculate similarity
        for intent in self.df['intent'].unique():
            # Get embeddings of all questions in this intent
            intent_embeddings = np.vstack(self.df[self.df['intent'] == intent]['embedding'].values)

            # Calculate similarity between the user input and each embedding in the intent
            similarities = np.array([self.calculate_similarity(user_embedding, embedding) for embedding in intent_embeddings])
            avg_similarity = np.mean(similarities)

            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                detected_intent = intent

        return detected_intent, max_similarity


if __name__ == '__main__':
    intent_recognizer = IntentRecognizer()

    user_input = "Can you tell me about the courses I can take?"
    detected_intent, similarity_score = intent_recognizer.classify_intent(user_input)

    print(f"Detected intent: {detected_intent} (Similarity: {similarity_score:.4f})")

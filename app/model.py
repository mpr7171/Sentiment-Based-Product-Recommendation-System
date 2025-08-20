import nltk
import pandas as pd
import numpy as np
import re
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

class SentimentRecommenderModel:
    """
    A class to manage sentiment-based product recommendations for the Ebuss e-commerce platform.
    Loads pre-trained models and data, provides methods for recommendation and sentiment analysis.
    """
    
    ROOT_PATH = "models/"
    MODEL_NAME = "best_sentiment_model.pkl"
    VECTORIZER = "tfidf.pkl"
    RECOMMENDER = "best_recommendation_model.pkl"
    CLEANED_DATA = "cleaned_data.pkl"
    SENTIMENT_CACHE = "sentiment_cache.pkl"

    def __init__(self):
        """
        Initialize the SentimentRecommenderModel with pre-trained models and data.

        Loads the sentiment model, TF-IDF vectorizer, recommendation matrix, and dataset.
        Initializes NLTK lemmatizer and stopwords. Loads or initializes a sentiment cache.

        Raises:
            FileNotFoundError: If any pickle file or dataset is missing.
            pickle.UnpicklingError: If pickle files are corrupted or incompatible.
            ValueError: If loaded objects are of incorrect type or structure.
        """
        self.ensure_nltk_resources()
        
        try:
            self.model = pickle.load(open(self.ROOT_PATH + self.MODEL_NAME, 'rb'))
            if not hasattr(self.model, 'predict'):
                raise ValueError(f"Loaded model from {self.MODEL_NAME} is not a valid classifier.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {self.MODEL_NAME} not found in {self.ROOT_PATH}")
        except pickle.UnpicklingError:
            raise ValueError(f"Error unpickling {self.MODEL_NAME}. File may be corrupted.")

        try:
            self.vectorizer = pickle.load(open(self.ROOT_PATH + self.VECTORIZER, 'rb'))
            if not hasattr(self.vectorizer, 'transform'):
                raise ValueError(f"Loaded vectorizer from {self.VECTORIZER} is not a valid vectorizer.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Vectorizer file {self.VECTORIZER} not found in {self.ROOT_PATH}")
        except pickle.UnpicklingError:
            raise ValueError(f"Error unpickling {self.VECTORIZER}. File may be corrupted.")

        try:
            self.user_final_rating = pickle.load(open(self.ROOT_PATH + self.RECOMMENDER, 'rb'))
            if not isinstance(self.user_final_rating, pd.DataFrame):
                raise ValueError(f"Loaded recommender from {self.RECOMMENDER} is not a pandas DataFrame.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Recommender file {self.RECOMMENDER} not found in {self.ROOT_PATH}")
        except pickle.UnpicklingError:
            raise ValueError(f"Error unpickling {self.RECOMMENDER}. File may be corrupted.")

        try:
            self.cleaned_data = pickle.load(open(self.ROOT_PATH + self.CLEANED_DATA, 'rb'))
            if not isinstance(self.cleaned_data, pd.DataFrame):
                raise ValueError(f"Loaded cleaned data from {self.CLEANED_DATA} is not a pandas DataFrame.")
            required_columns = ['id', 'name', 'reviews_cleaned']
            if not all(col in self.cleaned_data.columns for col in required_columns):
                raise ValueError(f"Cleaned data missing required columns: {required_columns}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Cleaned data file {self.CLEANED_DATA} not found in {self.ROOT_PATH}")
        except pickle.UnpicklingError:
            raise ValueError(f"Error unpickling {self.CLEANED_DATA}. File may be corrupted.")

        try:
            self.data = pd.read_csv("data/sample30.csv")
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError("Loaded dataset from 'data/sample30.csv' is not a pandas DataFrame.")
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file 'data/sample30.csv' not found.")

        try:
            self.sentiment_cache = pickle.load(open(self.ROOT_PATH + self.SENTIMENT_CACHE, 'rb'))
            if not isinstance(self.sentiment_cache, dict):
                raise ValueError(f"Loaded sentiment cache from {self.SENTIMENT_CACHE} is not a dictionary.")
        except (FileNotFoundError, pickle.UnpicklingError):
            self.sentiment_cache = {}

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {"not", "never", "no"}

    def ensure_nltk_resources(self):
        """
        Ensure required NLTK resources are available, downloading if necessary.

        Resources checked: stopwords, punkt, averaged_perceptron_tagger, wordnet, omw-1.4.
        """
        resources = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

    def save_sentiment_cache(self):
        """
        Save the sentiment cache to disk for reuse across sessions.

        Raises:
            IOError: If writing to the cache file fails.
        """
        try:
            with open(self.ROOT_PATH + self.SENTIMENT_CACHE, 'wb') as f:
                pickle.dump(self.sentiment_cache, f)
        except IOError as e:
            print(f"Warning: Failed to save sentiment cache to {self.SENTIMENT_CACHE}: {e}")

    def getRecommendationByUser(self, user):
        """
        Get top 20 recommended product IDs from collaborative filtering.

        Args:
            user (str): Username to generate recommendations for.

        Returns:
            list: List of top 20 product IDs sorted by predicted rating.
                  Returns None if user does not exist.

        Raises:
            KeyError: If user_final_rating DataFrame is malformed.
        """
        try:
            if user in self.user_final_rating.index:
                return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            else:
                print(f"User name '{user}' doesn't exist")
                return None
        except KeyError as e:
            raise KeyError(f"Error accessing user_final_rating for user '{user}': {e}")

    def getSentimentRecommendations(self, user):
        """
        Filter top 5 positive products from top 20 recommendations based on sentiment.

        Args:
            user (str): Username to generate recommendations for.

        Returns:
            pd.DataFrame: DataFrame with columns 'name', 'brand', 'manufacturer',
                          'pos_sentiment_percent' for the top 5 products, sorted by
                          positive sentiment percentage and name. Returns None if user
                          does not exist.

        Raises:
            KeyError: If required columns are missing in the dataset.
            ValueError: If sentiment prediction fails due to invalid data.
        """
        try:
            if user not in self.user_final_rating.index:
                print(f"User name '{user}' doesn't exist")
                return None
            
            recommendations = list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            results = []
            
            # Process each product, using cache if available
            for prod_id in recommendations:
                if prod_id in self.sentiment_cache:
                    results.append(self.sentiment_cache[prod_id])
                    continue
                
                filtered_data = self.cleaned_data[self.cleaned_data.id == prod_id].copy()
                if filtered_data.empty:
                    continue
                
                # Predict sentiment for reviews
                X = self.vectorizer.transform(filtered_data["reviews_cleaned"].values.astype(str))
                filtered_data["predicted_sentiment"] = self.model.predict(X)
                
                # Calculate positive sentiment percentage
                pos_count = filtered_data["predicted_sentiment"].sum()
                total_count = len(filtered_data)
                pos_percent = np.round(pos_count / total_count * 100, 2) if total_count > 0 else 0
                
                # Cache result
                self.sentiment_cache[prod_id] = {
                    'name': filtered_data['name'].iloc[0],
                    'pos_sentiment_percent': pos_percent
                }
                results.append(self.sentiment_cache[prod_id])
            
            # Save cache to disk
            self.save_sentiment_cache()
            
            # Convert results to DataFrame and merge with additional product info
            result_df = pd.DataFrame(results).sort_values(
                ['pos_sentiment_percent', 'name'], ascending=[False, True])[0:5]
            if result_df.empty:
                print(f"No valid recommendations found for user '{user}'")
                return None
            
            return pd.merge(self.data, result_df, on='name')[
                ['name', 'brand', 'manufacturer', 'pos_sentiment_percent']
            ].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        
        except KeyError as e:
            raise KeyError(f"Error processing recommendations: {e}")
        except Exception as e:
            raise ValueError(f"Error in sentiment prediction or data processing: {e}")


    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for text data.

        Args:
            text (str): Raw text to preprocess.

        Returns:
            str: Preprocessed text (lowercase, cleaned, stopword-removed, lemmatized).

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text

    def clean_text(self, text):
        """
        Clean text by converting to lowercase and removing punctuation, brackets, numbers.

        Args:
            text (str): Raw text to clean.

        Returns:
            str: Cleaned text.
        """
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)  # Remove [bracketed] content
        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def remove_stopwords(self, text):
        """
        Remove stopwords from text, retaining negation words (not, never, no).

        Args:
            text (str): Text to process.

        Returns:
            str: Text with stopwords removed.
        """
        return " ".join([word for word in text.split() if word.isalpha() and word not in self.stop_words])

    def lemmatize_text(self, text):
        """
        Lemmatize text using POS tagging for accurate normalization.

        Args:
            text (str): Text to lemmatize.

        Returns:
            str: Lemmatized text.
        """
        words_pos = nltk.pos_tag(word_tokenize(text))
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in words_pos
        ]
        return " ".join(lemmatized)

    def get_wordnet_pos(self, tag):
        """
        Map POS tag to format accepted by WordNetLemmatizer.

        Args:
            tag (str): POS tag from nltk.pos_tag.

        Returns:
            str: WordNet POS tag (NOUN, VERB, ADJ, ADV).
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
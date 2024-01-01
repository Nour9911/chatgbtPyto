import os
import json
import random
import numpy as np
import nltk
import tensorflow as tf
from neuralintents.assistants import BasicAssistant
from neuralintents.assistants import GenericAssistant

class BasicAssistant:
    def __init__(self, intents_data):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

        if isinstance(intents_data, dict):
            self.intents_data = intents_data
        else:
            if os.path.exists(intents_data):
                with open(intents_data, "r") as f:
                    self.intents_data = json.load(f)
            else:
                raise FileNotFoundError

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.words = []
        self.intents = []
        self.training_data = []

    def _prepare_intents_data(self, ignore_letters=("?","!",".",",")):
        documents = []

        for intent in self.intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])

            for pattern in intent["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                self.words += pattern_words
                documents.append((pattern_words, intent["tag"]))

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(set(self.words))

        empty_output = [0] * len(self.intents)

        for document in documents:
            bag_of_words = []
            pattern_words = document[0]
            pattern_words = [self.lemmatizer.lemmatize(w.lower()) for w in pattern_words]
            for word in self.words:
                bag_of_words.append(1 if word in pattern_words else 0)

            output_row = empty_output.copy()
            output_row[self.intents.index(document[1])] = 1
            self.training_data.append([bag_of_words, output_row])

        random.shuffle(self.training_data)
        self.training_data = np.array(self.training_data, dtype="object")

        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])

        return X, y

    def fit_model(self, optimizer=None, epochs=200):
        X, y = self._prepare_intents_data()

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(y.shape[1], activation='softmax')
        ])

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        history = model.fit(X, y, epochs=epochs, batch_size=5, verbose=1)

        return model, history

    def save_model(self, model, words, intents, model_name="basic_model"):
        model.save(f"{model_name}.keras")
        with open(f'{model_name}_words.pkl', 'wb') as f:
            pickle.dump(words, f)
        with open(f'{model_name}_intents.pkl', 'wb') as f:
            pickle.dump(intents, f)

    def load_model(self, model_name="basic_model"):
        model = tf.keras.models.load_model(f'{model_name}.keras')
        with open(f'{model_name}_words.pkl', 'rb') as f:
            words = pickle.load(f)
        with open(f'{model_name}_intents.pkl', 'rb') as f:
            intents = pickle.load(f)

        return model, words, intents

    def _predict_intent(self, model, input_text):
        input_words = nltk.word_tokenize(input_text)
        input_words = [self.lemmatizer.lemmatize(w.lower()) for w in input_words]

        input_bag_of_words = [0] * len(self.words)

        for input_word in input_words:
            for i, word in enumerate(self.words):
                if input_word == word:
                    input_bag_of_words[i] = 1

        input_bag_of_words = np.array([input_bag_of_words])

        predictions = model.predict(input_bag_of_words, verbose=0)[0]
        predicted_intent = self.intents[np.argmax(predictions)]

        return predicted_intent

    def process_input(self, model, input_text):
        predicted_intent = self._predict_intent(model, input_text)

        for intent in self.intents_data["intents"]:
            if intent["tag"] == predicted_intent:
                return random.choice(intent["responses"])

        return "I don't understand that."
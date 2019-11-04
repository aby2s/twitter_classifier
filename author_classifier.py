import datetime
import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy
import spacy
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from spacy.lang.en import English


class AuthorDataset:
    AUTHORS = ['Neil deGrasse Tyson', 'Cristiano Ronaldo', 'Ellen DeGeneres', 'Sebastian Ruder',
               'KATY PERRY', 'Kim Kardashian West', 'Snoop Dogg', 'Elon Musk', 'Barack Obama',
               'Donald J. Trump']

    LANGS = ['en', 'pt', 'und', 'tl', 'ht', 'in', 'sv', 'es', 'it', 'pl', 'hi', 'et', 'ca', 'de', 'fi', 'fr',
             'ru', 'hu', 'nl', 'da', 'tr', 'lt', 'ro', 'eu', 'ja', 'cy', 'sl', 'is', 'no', 'cs']

    DAYS_OF_WEEK = ['Mon', 'Sat', 'Fri', 'Wed', 'Tue', 'Sun', 'Thu']

    def __init__(self, path_to_csv):
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self._nlp = English()

        self._df = pd.read_csv(path_to_csv, encoding='utf8', engine='python')

        self._class_mappings = {n: i for i, n in enumerate(self.AUTHORS)}
        self._lang_mappings = {n: i for i, n in enumerate(self.LANGS)}
        self._dow_mappings = {n: i for i, n in enumerate(self.DAYS_OF_WEEK)}

    def get_labels(self):
        return self._df['author'].apply(lambda x: self._class_mappings[x]).values

    def get_metadata(self):
        return np.vstack([self._df['source'],
                          self._df['lang'].apply(lambda x: self._lang_mappings[x]).values,
                          self._df['day_of_week'].apply(lambda x: self._dow_mappings[x]).values,
                          self._df['has_hashtag'].astype(int).values,
                          self._df['has_mentions'].astype(int).values,
                          self._df['has_url'].astype(int).values,
                          self._df['has_media'].astype(int).values,
                          self._df['day'].values, self._df['month'].values,
                          self._df['year'].values, self._df['hour'].values, self._df['minute'].values,
                          self._df['second'].values, self._df['day_of_year'].values,
                          self._df['week_of_year'].values]).T

    def get_tweets(self):
        return self._df['tweet'].apply(
            lambda x: ' '.join([w.lemma_ for w in self._nlp(x) if w.is_stop == False])).values

    def fill_predictions(self, predictions):
        self._df['author'] = [self.AUTHORS[v] for v in predictions]

    def to_csv(self, path_to_csv):
        self._df.to_csv(path_to_csv, index=False)


class AuthorClassifier:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()

    def fit(self, tweets, metadata, labels, test_size):
        print('Model training is in progress')

        if test_size is not None:
            split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_index, test_index = next(split.split(tweets, labels))

            tweets_train, tweets_test = tweets[train_index], tweets[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            metadata_train, metadata_test = metadata[train_index], metadata[test_index]

            tweets_train = self._vectorizer.fit_transform(tweets_train)
            tweets_test = self._vectorizer.transform(tweets_test)

            X_train = scipy.sparse.hstack([tweets_train, metadata_train])
            X_test = scipy.sparse.hstack([tweets_test, metadata_test])

            self._fit(X_train, labels_train)

            train_predictions = self._predict(X_train)
            print('Train set score {}'.format(precision_score(labels_train, train_predictions, average='micro')))

            test_predictions = self._predict(X_test)
            print('Test set score {}'.format(precision_score(labels_test, test_predictions, average='micro')))

        else:
            tweets_train = self._vectorizer.fit_transform(tweets)
            X_train = scipy.sparse.hstack([tweets_train, metadata])

            self._fit(X_train, labels)

            train_predictions = self._predict(X_train)
            print('Train set score {}'.format(precision_score(labels, train_predictions, average='micro')))

    def kfold_validation(self, tweets, metadata, labels, nsplits=5):
        skf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=42)

        prs = list()

        for idx, (train_index, test_index) in enumerate(skf.split(metadata, labels)):
            print('K-Fold iteration {}'.format(idx + 1))

            tweets_train, tweets_test = tweets[train_index], tweets[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            metadata_train, metadata_test = metadata[train_index], metadata[test_index]

            self._initialize()
            self.fit(tweets_train, metadata_train, labels_train, test_size=None)
            test_predictions = self.predict(tweets_test, metadata_test)
            pr = precision_score(labels_test, test_predictions, average='micro')
            prs.append(pr)
            print('Test set score {}'.format(pr))

        print('KFold test results {}'.format(prs))

    def predict(self, tweets, metadata):
        tweets = self._vectorizer.transform(tweets)
        X_train = scipy.sparse.hstack([tweets, metadata])

        return self._predict(X_train)

    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass

    def _initialize(self):
        pass


class RFClassifier(AuthorClassifier):
    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self):
        self._model = RandomForestClassifier(n_estimators=1000)  # n_estimators=1000, max_depth=50)

    def _fit(self, X, y):
        self._model.fit(X, y)

    def _predict(self, X):
        return self._model.predict(X)

    def dump(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, 'model.pickle'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def restore(self, model_path):
        with open(os.path.join(model_path, 'model.pickle'), 'rb') as f:
            return pickle.load(f)


class CBClassifier(AuthorClassifier):
    def __init__(self):
        super().__init__()
        self._initialize()

    def _initialize(self):
        self._model = CatBoostClassifier(learning_rate=0.3, depth=10, l2_leaf_reg=0.1, task_type="GPU")

    def _fit(self, X, y):
        self._model.fit(X, y)

    def _predict(self, X):
        return np.reshape(self._model.predict(X).astype(int), (-1,))

    def dump(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self._model.save_model(os.path.join(model_path, './model.cbm'))

        with open(os.path.join(model_path, 'vectorizer.pickle'), 'wb') as f:
            pickle.dump(self._vectorizer, f)

    @classmethod
    def restore(self, model_path):
        res = CBClassifier()
        res._model.load_model(os.path.join(model_path, './model.cbm'))
        with open(os.path.join(model_path, 'vectorizer.pickle'), 'rb') as f:
            res._vectorizer = pickle.load(f)

        return res


def main():
    parser = ArgumentParser()

    parser.add_argument('--task', type=str, required=True, choices=['fit', 'kfold', 'evaluation'],
                        help='Task: fit, kfold validation or predict')
    parser.add_argument('--model_path', type=str, required=True, help='Path to a model serialization directory')
    parser.add_argument('--ds_path', type=str, required=True, help='Path to a dataset')
    parser.add_argument('--test_size', type=float, required=False, help='Test split size for fit task')
    parser.add_argument('--nsplits', type=int, required=False, help='Number of splits for KFold validation', default=5)
    parser.add_argument('--eval_path', type=str, required=False, help='Path to an evaluation results csv')

    args = parser.parse_args()

    dataset = AuthorDataset(args.ds_path)
    tweets = dataset.get_tweets()
    metadata = dataset.get_metadata()

    if args.task == 'fit':
        labels = dataset.get_labels()
        model = CBClassifier()
        model.fit(tweets, metadata, labels, test_size=args.test_size)
        model.dump(args.model_path)
    elif args.task == 'kfold':
        labels = dataset.get_labels()
        model = CBClassifier()
        model.kfold_validation(tweets, metadata, labels, nsplits=args.nsplits)
        model.dump(args.model_path)
    elif args.task == 'evaluation':
        model = CBClassifier.restore(args.model_path)
        tic = datetime.datetime.now()
        predictions = model.predict(tweets, metadata)
        toc = datetime.datetime.now()
        print('Prediction time: {:.2f} seconds'.format((toc - tic).total_seconds()))
        dataset.fill_predictions(predictions)
        dataset.to_csv(args.eval_path)


if __name__ == "__main__":
    sys.exit(main())

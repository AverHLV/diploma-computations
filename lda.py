import pandas as pd
from numpy import argmax, array
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyLDAvis import show
from pyLDAvis.sklearn import prepare
from collections import Counter
from pickle import load, dump
from json import loads


class Corpora(object):
    """ Managing text collections class for use in LDA algorithm """

    def __init__(self, clear=False, filename='csv/topics.csv', filename_for_save=None,
                 stopwords_filename='json/stopwords.json', tf_idf_thresholds=(2, 7), min_string_length=10):
        """
        Corpora class initialization

        :param clear: clear loaded corpora or not
        :param filename: csv filename for load
        :param filename_for_save: filename with csv format for saving corpora
        :param stopwords_filename: stopwords filename in json format
        :param tf_idf_thresholds: pair of lower and upper thresholds for clearing by tf-idf value
        :param min_string_length: minimum possible string length in corpora
        """

        self.filename_for_save = filename_for_save

        # load data

        self.stopwords = self.load_stopwords(stopwords_filename)
        self.__corpora = pd.read_csv(filename)['topic'].tolist()

        # clear corpora

        if clear:
            self.clean_spaces()
            self.delete_words()
            self.tf_idf(tf_idf_thresholds)
            self.delete_small_strings(min_string_length)

        # save results

        self.save_corpora()

    @property
    def corpora(self):
        return self.__corpora

    def clean_spaces(self):
        for i in range(len(self.corpora)):
            if self.corpora[i][0] == ' ':
                self.corpora[i] = self.corpora[i][1:]

            if self.corpora[i][-1] == ' ':
                self.corpora[i] = self.corpora[i][:-1]

    def delete_words(self, words=None):
        """ Delete specific words or stopwords from corpora """

        if words is None:
            self.__corpora = [
                ' '.join([word for word in string.split(' ') if word not in self.stopwords])
                for string in self.__corpora
            ]

        else:
            self.__corpora = [
                ' '.join([word for word in string.split(' ') if word not in words]) for string in self.__corpora
            ]

    def tf_idf(self, thresholds):
        """ Delete uninformative words by TF-IDF value """

        if thresholds[0] >= thresholds[1]:
            raise ValueError('Thresholds[0] must be lower than thresholds[1]')

        vectorizer = TfidfVectorizer(lowercase=False)
        vectorizer.fit(self.__corpora)

        features = vectorizer.get_feature_names()
        values = vectorizer.idf_
        words = [features[i] for i in range(len(values)) if values[i] < thresholds[0] or values[i] > thresholds[1]]

        if len(words):
            self.delete_words(words)

    def delete_small_strings(self, min_string_length):
        """ Delete very small strings from corpora """

        self.__corpora = [string for string in self.__corpora if len(string) >= min_string_length]

    def words_freq(self, number=10):
        """ Display %number% most common words in corpora """

        words_full_list = []

        for string in self.__corpora:
            words_full_list += string.split()

        print(Counter(words_full_list).most_common(number))

    def save_corpora(self):
        """ Save corpora to csv """

        if self.filename_for_save is not None:
            pd.DataFrame(self.corpora, columns=['topic']).to_csv(self.filename_for_save, index=False)

    @staticmethod
    def load_stopwords(filename):
        """ Load stopwords in json format """

        with open(filename, encoding='utf8') as file:
            return loads(file.read())['stop_words']


class LDA(object):
    """ Class for Latent Dirichlet Allocation model """

    def __init__(self, filename='models/lda.model'):
        self.filename = filename
        self.vectorized_data = None
        self.df_topic_keywords = None
        self.fitted = False

        # load fitted model if exists

        try:
            self.load_model(self.filename)
            self.fitted = True

        except IOError:
            self.vectorizer = CountVectorizer(lowercase=False)
            self.lda = LatentDirichletAllocation()

    def fit(self, corpora):
        self.vectorized_data = self.vectorizer.fit_transform(corpora)

        search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]}
        model = GridSearchCV(self.lda, param_grid=search_params, cv=5)
        model.fit(self.vectorized_data)

        self.lda = model.best_estimator_
        self.fitted = True
        self.construct_df_topics()
        self.save_model()

    def predict(self, text):
        if not self.fitted:
            raise ValueError('LDA model is not fitted')

        topic_probability_scores = self.lda.transform(self.vectorizer.transform(text))[0]
        topics = self.df_topic_keywords.iloc[argmax(topic_probability_scores), :].values.tolist()
        topics = list(zip(topics, topic_probability_scores))
        return sorted(topics, key=lambda x: x[1], reverse=True)

    def construct_df_topics(self, n_words=20):
        """ Construct pd.DataFrame with top n keywords for each topic """

        if not self.fitted:
            raise ValueError('LDA model is not fitted')

        topic_keywords = []
        keywords = array(self.vectorizer.get_feature_names())

        for topic_weights in self.lda.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))

        self.df_topic_keywords = pd.DataFrame(topic_keywords)
        self.df_topic_keywords.columns = ['Word ' + str(i) for i in range(self.df_topic_keywords.shape[1])]
        self.df_topic_keywords.index = ['Topic ' + str(i) for i in range(self.df_topic_keywords.shape[0])]

    def stats(self):
        if not self.fitted:
            raise ValueError('LDA model is not fitted')

        print('Log Likelihood:', self.lda.score(self.vectorized_data))
        print('Perplexity:', self.lda.perplexity(self.vectorized_data))

    def visualize(self):
        """ Start local web-server to display the LDA fitted model """

        if not self.fitted:
            raise ValueError('LDA model is not fitted')

        show(prepare(self.lda, self.vectorized_data, self.vectorizer, mds='tsne'))

    def load_model(self, filename):
        """ Load LDA model, CountVectorizer instance and term-document matrix from binary file """

        with open(filename, 'rb') as file:
            model_dict = load(file)

        self.lda = model_dict['lda']
        self.vectorizer = model_dict['vec']
        self.vectorized_data = model_dict['vec_data']
        self.df_topic_keywords = model_dict['df']

    def save_model(self):
        """ Save fitted LDA model by pickle """

        if not self.fitted:
            print('LDA model is not fitted')
            return

        with open(self.filename, 'wb') as file:
            dump({'lda': self.lda, 'vec': self.vectorizer, 'vec_data': self.vectorized_data,
                  'df': self.df_topic_keywords}, file)


if __name__ == '__main__':
    lda = LDA()
    lda.visualize()

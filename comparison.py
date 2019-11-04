import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import system
from pathlib import Path
from pickle import load, dump

from seaborn import heatmap, violinplot

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier


def set_logger(logger_name: str = 'comparison') -> logging.Logger:
    """ Create and configure a logger """

    logging.basicConfig(
        format='%(filename)s[%(levelname)s][%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=logger_name + '.log'
    )

    return logging.getLogger(logger_name)


logger = set_logger()


def load_data(path: Path, info: bool = False) -> pd.DataFrame:
    """ Load pd.DataFrame from csv """

    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])

    if info:
        print('Columns:', df.columns, '\nShape:', df.shape)

    return df


def visualize(df: pd.DataFrame, descriptors: tuple, show: str = 'heat') -> None:
    """
    Visualise different statistics from pd.DataFrame

    :param df: pd.DataFrame
    :param descriptors: pair of column indices descriptors in DataFrame
    :param show: what plot to show
    """

    assert show in ('heat', 'hist', 'violin'), f'Wrong "show" value: {show}.'

    if show == 'heat':
        # calculate features heatmap

        corr = df[df.columns[descriptors[0]:descriptors[1]]].corr()
        heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
        plt.title('Features heatmap')
        plt.show()

        # calculate mean class-features heatmap

        corr = {column: [] for column in df.columns[descriptors[0]:descriptors[1]]}

        for column in df[df.columns[descriptors[1]:]]:
            for descriptor in corr:
                corr[descriptor].append(df[column].corr(df[descriptor]))

        for descriptor in corr:
            corr[descriptor] = np.mean(corr[descriptor])

        corr = pd.DataFrame(corr.values(), columns=['Class'], index=corr.keys())
        heatmap(corr, xticklabels=corr.columns, yticklabels=corr.index, annot=True)
        plt.title('Mean class-features heatmap')
        plt.show()

    elif show == 'hist':
        for column in df.columns[descriptors[0]:descriptors[1]]:
            plt.hist(df[column])
            plt.title(column + ' hist')
            plt.show()

    else:
        for column in df.columns[descriptors[0]:descriptors[1]]:
            violinplot(df[column])
            plt.title(column + ' violinplot')
            plt.show()


def scale(df, descriptors):
    """ Scale numeric fields to [0, 1] """

    for column in df.columns[descriptors[0]:descriptors[1]]:
        df[column] = MinMaxScaler().fit_transform(df[column].values.reshape(-1, 1))

    return df


def split(df, n_splits=5):
    """ Split pd.DataFrame to train / test subsets for cross-validation """

    return KFold(n_splits=n_splits).split(df)


@metrics.make_scorer
def accuracy(real, prediction):
    """
    Calculate average accuracy for real and predicted labels

    :param real: np.array
    :param prediction: np.array
    :return: float
    """

    return np.mean(np.array([metrics.accuracy_score(real[:, i], prediction[:, i]) for i in range(prediction.shape[1])]))


class BaseClassifier(object):
    """
    Base classifiers class that provides saving / loading model objects
    and fit / predict methods
    """

    search_params = {}

    def __init__(self, input_data, descriptors, filename_for_save):
        self.model = None
        self.fitted = False
        self.input_data = input_data
        self.descriptors = descriptors
        self.filename_for_save = filename_for_save
        self.load_model()

    def __str__(self):
        print_string = 'Classifier object. Params:\n'
        params = self.model.get_params()

        for key in params:
            print_string += '{0}: {1}\n'.format(key, params[key])

        return print_string

    def check_model(self):
        if not self.fitted or self.model is None:
            raise ValueError('Model is not fitted or not created')

    def fit(self):
        """
        Fit classifier with 'train' data, find best hyper parameters via grid search
        and save best model
        """

        model = GridSearchCV(self.model, param_grid=self.search_params, scoring=accuracy, cv=3, iid=True)

        model.fit(
            self.input_data['train'][self.input_data['train'].columns[self.descriptors[0]:self.descriptors[1]]].values,
            self.input_data['train'][self.input_data['train'].columns[self.descriptors[1]:]].values
        )

        self.model = model.best_estimator_
        self.fitted = True

        # save fitted model and found parameters

        self.save_model()
        self.save_model_params()

    def predict(self):
        """
        Predict class labels for 'test' data and
        return classification score by accuracy scorer
        """

        self.check_model()

        return accuracy(
            self.model,
            self.input_data['test'][self.input_data['test'].columns[self.descriptors[0]:self.descriptors[1]]].values,
            self.input_data['test'][self.input_data['test'].columns[self.descriptors[1]:]].values
        )

    def save_model(self):
        """ Save sklearn model to binary file by pickle """

        self.check_model()

        with open(self.filename_for_save, 'wb') as file:
            dump(self.model, file)

    def save_model_params(self):
        """ Save sklearn model parameters to file """

        self.check_model()

        with open(self.filename_for_save + '.params', 'w') as file:
            file.write(self.__str__())

    def load_model(self):
        """ Load sklearn model from binary file by pickle """

        try:
            with open(self.filename_for_save, 'rb') as file:
                self.model = load(file)
                self.fitted = True

        except IOError:
            pass


class KNeighbors(BaseClassifier):
    search_params = {
        'n_neighbors': [4, 5, 6, 7, 8],
        'p': [1, 2, 3, 4]
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = KNeighborsClassifier()


class DecisionTree(BaseClassifier):
    search_params = {
        'max_depth': [5, 7, 9, 11],
        'criterion': ['gini', 'entropy']
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = DecisionTreeClassifier()

    def save_to_dot(self):
        """ Save built decision tree in dot format """

        self.check_model()

        export_graphviz(
            self.model, filled=True, out_file=self.filename_for_save + '.dot',
            feature_names=self.input_data['test'].columns[self.descriptors[0]:self.descriptors[1]]
        )


class RandomForest(BaseClassifier):
    search_params = {
        'n_estimators': [100, 130, 170, 200],
        'max_depth': [5, 7, 10],
        'criterion': ['gini', 'entropy']
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = RandomForestClassifier()

    def save_feature_importances(self, column_names_max_length=5):
        """ Save feature importances dictionary and bar diagram to files """

        self.check_model()

        indices = np.argsort(self.model.feature_importances_)[::-1]
        importance_values = self.model.feature_importances_[indices]

        st_dev = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        st_dev = st_dev[indices]

        column_names = [
            column if len(column) <= column_names_max_length else column[:column_names_max_length]
            for column in self.input_data['test'].columns[self.descriptors[0]:self.descriptors[1]][indices]
        ]

        importances = [(column, importance_values[index]) for index, column in enumerate(column_names)]
        importances = sorted(importances, reverse=True, key=lambda x: x[1])

        # save feature importances dictionary to json

        with open(self.filename_for_save + '.imp', 'w', encoding='utf8') as file:
            for feature in importances:
                file.write('{0}: {1}\n'.format(*feature))

        # create and save feature importances diagram

        bar_numbers = range(len(importance_values))

        plt.bar(bar_numbers, importance_values, yerr=st_dev, align='center')
        plt.title('Feature importances from ' + self.filename_for_save)
        plt.xticks(bar_numbers, column_names, rotation=25)
        plt.savefig(self.filename_for_save + '.png')
        plt.clf()


class NeuralNetwork(BaseClassifier):
    search_params = {
        'estimator__hidden_layer_sizes': [(100, 100)],
        'estimator__activation': ['logistic', 'tanh', 'relu'],
        'estimator__solver': ['lbfgs', 'sgd', 'adam']
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = MultiOutputClassifier(MLPClassifier(), n_jobs=6)


def dot_to_png(path: Path) -> None:
    """
    Convert all .dot files from specified folder to png

    :param path: path to folder with .dot files
    """

    for file in path.glob('*.dot'):
        system(f'dot -Tpng {file} -o {str(file)[:-3] + "png"}')


def compare(df, descriptors):
    """ Compare ML algorithms by cross-validation strategy """

    iteration = 0
    winners = {'nb': 0, 'dt': 0, 'rf': 0, 'nn': 0}
    logger.info('Starting...')

    for train, test in split(df):
        current_scores = {}
        learning_data = {'train': df.loc[train], 'test': df.loc[test]}

        # neighbors = KNeighbors(learning_data, descriptors, 'models/nb-{0}.model'.format(iteration))
        # neighbors.fit()
        # current_scores['nb'] = neighbors.predict()

        # des_tree = DecisionTree(learning_data, descriptors, 'models/dt-{0}.model'.format(iteration))
        # des_tree.fit()
        # des_tree.save_to_dot()
        # current_scores['dt'] = des_tree.predict()

        # forest = RandomForest(learning_data, descriptors, 'models/rf-{0}.model'.format(iteration))
        # forest.fit()
        # forest.save_feature_importances()
        # current_scores['rf'] = forest.predict()

        network = NeuralNetwork(learning_data, descriptors, 'models/nn-{0}.model'.format(iteration))
        network.fit()
        current_scores['nn'] = network.predict()

        current_scores = sorted(list(current_scores.items()), reverse=True, key=lambda x: x[1])
        winners[current_scores[0][0]] += 1

        logger.info(
            'Iter# {0}, winner: {1}, accuracy values: {2}'.format(iteration, current_scores[0][0], current_scores)
        )

        iteration += 1

    logger.info('Results: {0}'.format(winners))


if __name__ == '__main__':
    base_dir = Path.cwd()
    descriptors_index = 1, 11

    # dot_to_png(base_dir / 'models')

    data = load_data(base_dir / 'csv' / 'data.csv')
    # visualize(data, descriptors_index)
    # data = scale(data, descriptors_index)

    # compare(data, descriptors_index)

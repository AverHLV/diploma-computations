import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import system
from time import time
from pathlib import Path
from pickle import load, dump

from seaborn import heatmap, violinplot

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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

    assert show in ('stats', 'class', 'heat', 'hist', 'violin'), f'Wrong "show" value: {show}.'

    if show == 'stats':
        for column in df.columns[descriptors[0]:descriptors[1]]:
            print(df[column].describe(), '\n')

    elif show == 'class':
        frequencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        for column in df.columns[descriptors[1]:]:
            print(f'\n{column}')
            counts = dict(df[column].value_counts())
            print(counts)

            for key in counts:
                frequencies[key] += counts[key]

        print('\nFrequencies:')
        print(frequencies)

    elif show == 'heat':
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


def scale(df: pd.DataFrame, descriptors: tuple) -> pd.DataFrame:
    """ Scale numeric fields by Standard scaler """

    df[df.columns[descriptors[0]:descriptors[1]]] = StandardScaler().fit_transform(
        df[df.columns[descriptors[0]:descriptors[1]]]
    )

    return df


def split(df: pd.DataFrame, n_splits: int = 5) -> tuple:
    """ Split pd.DataFrame to train / test subsets for cross-validation """

    return KFold(n_splits=n_splits, shuffle=True).split(df)


def data_representation(df: pd.DataFrame, descriptors: tuple, show: bool = True, components: tuple = (2, 3)) -> None:
    """ Build and save or show already built tsne models """

    for n_components in components:
        tsne = TSNEManifold(df, descriptors, f'models/tsne_{n_components}.model', n_components)

        if show:
            tsne.save_scatter(show=True)

        else:
            tsne.fit()


@metrics.make_scorer
def accuracy(real: np.ndarray, prediction: np.ndarray):
    """ Calculate average accuracy for real and predicted labels """

    return np.mean(np.array([metrics.accuracy_score(real[:, i], prediction[:, i]) for i in range(prediction.shape[1])]))


@metrics.make_scorer
def f1(real: np.ndarray, prediction: np.ndarray):
    """ Calculate average F1 measure for real and predicted labels """

    return np.mean(np.array(
        [metrics.f1_score(real[:, i], prediction[:, i], average='macro') for i in range(prediction.shape[1])]
    ))


class BaseClassifier(object):
    """
    Base classifiers class that provides saving / loading model objects
    and fit / predict methods
    """

    search_params = {}

    def __init__(self, input_data, descriptors: tuple, filename_for_save: str):
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
            print_string += f'{key}: {params[key]}\n'

        return print_string

    def check_model(self) -> None:
        if not self.fitted or self.model is None:
            raise ValueError('Model is not fitted or not created')

    def fit(self) -> None:
        """
        Fit classifier with 'train' data, find best hyper parameters via grid search
        and save best model
        """

        scoring = {'Accuracy': accuracy, 'F1': f1}

        model = GridSearchCV(
            self.model,
            param_grid=self.search_params,
            scoring=scoring,
            cv=3,
            n_jobs=6,
            pre_dispatch=6,
            iid=False,
            refit='Accuracy',
            error_score='raise',
            return_train_score=True
        )

        model.fit(
            self.input_data['train'][self.input_data['train'].columns[self.descriptors[0]:self.descriptors[1]]].values,
            self.input_data['train'][self.input_data['train'].columns[self.descriptors[1]:]].values
        )

        self.model = model.best_estimator_
        self.fitted = True

        # save fitted model and found parameters

        self.save_model()
        self.save_model_params()
        self.save_fit_plot(model.cv_results_, scoring)

    def predict(self) -> float:
        """
        Predict class labels for 'test' data and
        return classification score by metric scorer
        """

        self.check_model()

        return accuracy(
            self.model,
            self.input_data['test'][self.input_data['test'].columns[self.descriptors[0]:self.descriptors[1]]].values,
            self.input_data['test'][self.input_data['test'].columns[self.descriptors[1]:]].values
        )

    def save_model(self) -> None:
        """ Save sklearn model to binary file by pickle """

        self.check_model()

        with open(self.filename_for_save, 'wb') as file:
            dump(self.model, file)

    def save_model_params(self) -> None:
        """ Save sklearn model parameters to file """

        self.check_model()

        with open(self.filename_for_save + '.params', 'w') as file:
            file.write(self.__str__())

    def save_fit_plot(self, results: dict, scoring: dict) -> None:
        """ Build and save GridSearchCV metric dynamics """

        self.check_model()
        point_numbers = np.array(range(len(results['mean_test_Accuracy'])))

        for scorer in sorted(scoring):
            for sample, style in ('train', '--'), ('test', '-'):
                plt.plot(
                    point_numbers,
                    results[f'mean_{sample}_{scorer}'],
                    style,
                    alpha=1 if sample == 'test' else 0.7,
                    label=f'{scorer} ({sample})'
                )

            best_index = np.nonzero(results[f'rank_test_{scorer}'] == 1)[0][0]
            best_score = results[f'mean_test_{scorer}'][best_index]

            plt.plot([point_numbers[best_index], ] * 2, [0, best_score], linestyle='-.')
            plt.annotate('%0.2f' % best_score, (point_numbers[best_index], best_score + 0.0005))

        plt.title('GridSearchCV metrics dynamics for ' + self.filename_for_save)
        plt.xlabel('Number of observation')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(self.filename_for_save + '_metrics.png')
        plt.clf()

    def load_model(self) -> None:
        """ Load sklearn model from binary file by pickle """

        try:
            with open(self.filename_for_save, 'rb') as file:
                self.model = load(file)
                self.fitted = True

        except IOError:
            pass


class BaseManifold(BaseClassifier):
    """ Sklearn manifold base class """

    def __init__(self, input_data, descriptors, filename_for_save, n_components):
        super().__init__(input_data, descriptors, filename_for_save)

        assert n_components in (2, 3), f'Wrong n_components value: {n_components}'

        self.transformed = None
        self.n_components = n_components
        self.load_model()

    def __str__(self):
        print_string = 'Manifold object. Params:\n'
        params = self.model.get_params()

        for key in params:
            print_string += f'{key}: {params[key]}\n'

        return print_string

    def fit(self) -> None:
        columns = ['x0', 'x1'] if self.n_components == 2 else ['x0', 'x1', 'x2']

        self.transformed = pd.DataFrame(
            self.model.fit_transform(self.input_data[self.input_data.columns[self.descriptors[0]:self.descriptors[1]]]),
            columns=columns
        )

        self.fitted = True

        self.save_model()
        self.save_model_params()
        self.save_scatter()

    def save_scatter(self, title: str = 'Dataset representation', show: bool = False):
        self.check_model()

        if self.n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D

            ax = plt.axes(projection='3d')
            ax.scatter(self.transformed['x0'], self.transformed['x1'], self.transformed['x2'], '-b')
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_zlabel('x2')

        else:
            plt.scatter(self.transformed['x0'], self.transformed['x1'])
            plt.xlabel('x0')
            plt.ylabel('x1')

        plt.title(title)

        if not show:
            plt.savefig(self.filename_for_save + '.png')
            plt.clf()

        else:
            plt.show()

    def save_model(self) -> None:
        """ Save sklearn model to binary file by pickle """

        self.check_model()

        with open(self.filename_for_save, 'wb') as file:
            dump({'model': self.model, 'transformed': self.transformed}, file)

    def load_model(self) -> None:
        """ Load sklearn model from binary file by pickle """

        try:
            with open(self.filename_for_save, 'rb') as file:
                dictionary = load(file)
                self.model = dictionary['model']
                self.transformed = dictionary['transformed']
                self.fitted = True

        except IOError:
            pass


class TSNEManifold(BaseManifold):
    def __init__(self, input_data, descriptors, filename_for_save, n_components=2):
        super().__init__(input_data, descriptors, filename_for_save, n_components)

        if not self.fitted:
            self.model = TSNE(n_components=self.n_components)


class KNeighbors(BaseClassifier):
    search_params = {
        'n_neighbors': [20, 24, 28],
        'p': [1, 2, 3]
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = KNeighborsClassifier()


class DecisionTree(BaseClassifier):
    search_params = {
        'max_depth': [3, 5, 7]
    }

    def __init__(self, input_data, descriptors, filename_for_save):
        super().__init__(input_data, descriptors, filename_for_save)

        if not self.fitted:
            self.model = DecisionTreeClassifier()

    def save_to_dot(self):
        """ Save built decision tree in dot format """

        self.check_model()

        export_graphviz(
            self.model,
            filled=True,
            out_file=self.filename_for_save + '.dot',
            feature_names=self.input_data['test'].columns[self.descriptors[0]:self.descriptors[1]]
        )


class RandomForest(BaseClassifier):
    search_params = {
        'n_estimators': [60, 80, 100, 130, 180],
        'max_depth': [3, 5, 7]
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

        plt.bar(bar_numbers, importance_values, align='center')
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
    winners = {'nb': 0, 'dt': 0, 'rf': 0}
    logger.info('Starting...')

    for train, test in split(df):
        current_scores = {}
        learning_data = {'train': df.loc[train], 'test': df.loc[test]}

        # k-neighbors

        start_time = time()
        model_name = f'models/nb-{iteration}.model'
        neighbors = KNeighbors(learning_data, descriptors, model_name)
        neighbors.fit()
        current_scores['nb'] = neighbors.predict()
        logger.info(f'{model_name} fit time: {time() - start_time} s')

        # decision tree
        
        start_time = time()
        model_name = f'models/dt-{iteration}.model'
        des_tree = DecisionTree(learning_data, descriptors, model_name)
        des_tree.fit()
        des_tree.save_to_dot()
        current_scores['dt'] = des_tree.predict()
        logger.info(f'{model_name} fit time: {time() - start_time} s')

        # random forest
        
        start_time = time()
        model_name = f'models/rf-{iteration}.model'
        forest = RandomForest(learning_data, descriptors, model_name)
        forest.fit()
        forest.save_feature_importances()
        current_scores['rf'] = forest.predict()
        logger.info(f'{model_name} fit time: {time() - start_time} s')

        current_scores = sorted(list(current_scores.items()), reverse=True, key=lambda x: x[1])
        winners[current_scores[0][0]] += 1
        logger.info(f'Iter# {iteration}, winner: {current_scores[0][0]}, metric values: {current_scores}')
        iteration += 1

    logger.info(f'Results: {winners}')


if __name__ == '__main__':
    base_dir = Path.cwd()
    descriptors_index = 1, 11

    # dot_to_png(base_dir / 'models')

    data = load_data(base_dir / 'csv' / 'data.csv')

    # visualize(data, descriptors_index, show='class')
    data_representation(data, descriptors_index)

    # data = scale(data, descriptors_index)

    # compare(data, descriptors_index)

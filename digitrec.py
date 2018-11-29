import numpy as np
import pandas as pd
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings


class NeuralNetwork:

    warnings.filterwarnings("ignore")

    def __init__(self, params):

        self.train_data = None
        self.test_data = None

        if params['normalization'] is None:
            print("Normalization parameter cannot be none. Please provide a type of normalization")
            sys.exit(1)

        self._default_params = {'hidden_layer_sizes': (100, ), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
                               'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001,
                               'power_t': 0.5, 'max_iter': 200, 'shuffle':True, 'random_state':None, 'tol':0.0001,
                               'verbose': False, 'warm_start': False, 'momentum': 0.9, 'nesterovs_momentum': True,
                               'early_stopping': False, 'validation_fraction':0.1, 'beta_1': 0.9, 'beta_2': 0.999,
                               'epsilon': 1e-08, 'n_iter_no_change': 10}
        self.params = params
        self._model = None
        self._model = None
        self._x_train = None
        self._x_test = None
        self._y_train = None
        self._y_test = None
        self._params_overwrite()

    def _params_overwrite(self):

        try:

            for key in self._default_params:

                if key not in self.params.keys():
                    self.params[key] = self._default_params[key]

        except Exception as e:
            print('Please provide a valid dictionary as parameter i.e key value pair. e.g: '
                  '{"hidden_layer_sizes": (10, 10), "activation": "relu"}')

    @staticmethod
    def min_max_norm(dt_train, dt_test):

        """ Represents min max normalisation depending on the parameter selection"""

        print("Selected normalisation is scale between (-1, 1)")

        try:

            min_max_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            x_sub_train = min_max_scalar.fit_transform(dt_train)
            x_sub_test = min_max_scalar.transform(dt_test)
            return x_sub_train, x_sub_test

        except Exception as e:
            print("Normalization error -")
            print(e)
            sys.exit(1)

    @staticmethod
    def l2_norm(dt_train, dt_test):

        """ Represents l2 normalisation depending on the parameter selection"""

        print("Selected normalisation is l2")

        try:

            x_sub_train = preprocessing.normalize(dt_train, norm='l2')
            x_sub_test = preprocessing.normalize(dt_test, norm='l2')
            return x_sub_train, x_sub_test

        except Exception as e:
            print("Normalization error -")
            print(e)
            sys.exit(1)

    @staticmethod
    def l1_norm(dt_train, dt_test):

        """ Represents l1 normalisation depending on the parameter selection"""

        print("Selected normalisation is l1")

        try:

            x_sub_train = preprocessing.normalize(dt_train, norm='l1')
            x_sub_test = preprocessing.normalize(dt_test, norm='l1')
            return x_sub_train, x_sub_test

        except Exception as e:
            print("Normalization error -")
            print(e)
            sys.exit(1)

    @staticmethod
    def max_abs_scalar(dt_train, dt_test):

        """ Represents max absolute normalisation depending on the parameter selection"""

        print("Selected normalisation is max absolute scalar")

        try:
            max_abs = preprocessing.MaxAbsScaler(copy=True)
            x_sub_train = max_abs.fit_transform(dt_train)
            x_sub_test = max_abs.transform(dt_test)
            return x_sub_train, x_sub_test

        except Exception as e:
            print("Normalization error -")
            print(e)
            sys.exit(1)

    @staticmethod
    def robust_scalar(dt_train, dt_test):

        """ Represents robust scalar normalisation depending on the parameter selection """

        print("Selected normalisation is robust scalar")

        try:
            rbs_scalar = preprocessing.RobustScaler()
            x_sub_train = rbs_scalar.fit_transform(dt_train)
            x_sub_test = rbs_scalar.transform(dt_test)
            return x_sub_train, x_sub_test

        except Exception as e:
            print("Normalization error -")
            print(e)
            sys.exit(1)

    def _normalization_selection(self, x_train, x_test):

        """ This function selects the type of normalisation as encoded in the gene"""

        if self.params['normalization'] == "min_max_norm":
            return self.min_max_norm(x_train, x_test)
        elif self.params['normalization'] == "l2_norm":
            return self.l2_norm(x_train, x_test)
        elif self.params['normalization'] == "l1_norm":
            return self.l1_norm(x_train, x_test)
        elif self.params['normalization'] == "max_abs_scalar":
            return self.max_abs_scalar(x_train, x_test)
        elif self.params['normalization'] == "robust_scalar":
            return self.robust_scalar(x_train, x_test)
        else:
            print("Normalization selection error - Please provide valid normalisation "
                  "('min_max_norm', 'l2_norm', 'l1_norm', 'max_abs_scalar', 'robust_scalar')")
            sys.exit(1)

    def _data_pre_processing(self):

        # Shuffling the training data to ensure no bias in data arrangements

        try:

            self.train_data = self.train_data.sample(frac=1)
            y_train = np.asarray(self.train_data['labels'])
            self.train_data.drop('labels', axis=1, inplace=True)
            x_train = self.train_data.values

            y_test = np.asarray(self.test_data['labels'])
            self.test_data.drop('labels', axis=1, inplace=True)
            x_test = self.test_data.values

            x_train, x_test = self._normalization_selection(x_train, x_test)

            return x_train, y_train, x_test, y_test

        except Exception as e:
            print("Something wrong in data pre processing stage - ")
            print(e)
            sys.exit(1)

    def fit(self, train_data_path, test_data_path):

        print("Please wait while pre processing the data...")

        try:

            self.train_data = pd.read_csv(train_data_path)
            self.test_data = pd.read_csv(test_data_path)

        except FileNotFoundError:
            print("please provide a valid file path.")
            sys.exit(1)

        x_train, y_train, x_test, y_test = self._data_pre_processing()

        try:

            print("Please wait while model is training.... ")

            mlp = MLPClassifier(hidden_layer_sizes=self.params['hidden_layer_sizes'],
                                activation=self.params['activation'], solver=self.params['solver'],
                                alpha=self.params['alpha'], batch_size=self.params['batch_size'],
                                learning_rate=self.params['learning_rate'],
                                learning_rate_init=self.params['learning_rate_init'],
                                power_t=self.params['power_t'], max_iter=self.params['max_iter'],
                                shuffle=self.params['shuffle'], random_state=self.params['random_state'],
                                tol=self.params['tol'], verbose=self.params['verbose'],
                                warm_start=self.params['warm_start'], momentum=self.params['momentum'],
                                nesterovs_momentum=self.params['nesterovs_momentum'],
                                early_stopping=self.params['early_stopping'],
                                validation_fraction=self.params['validation_fraction'],
                                beta_1=self.params['beta_1'], beta_2=self.params['beta_2'],
                                epsilon=self.params['epsilon'], n_iter_no_change=self.params['n_iter_no_change'])
            mlp.fit(x_train, y_train)

            self._model = mlp
            self._x_train = x_train
            self._x_test = x_test
            self._y_train = y_train
            self._y_test = y_test

            y_pred = list(self._model.predict(self._x_train))

            print("\n\n ************************* Classification Report on Training Data ***************************\n")
            target_names = list(map(str, list(set(list(self._y_train)))))
            print(classification_report(list(self._y_train), y_pred, target_names=target_names))
            print("\n\n************************** Accuracy Score - Training Data ***********************************\n")
            print(accuracy_score(self._y_train, y_pred))

        except Exception as e:
            print("Something wrong with the parameter or data set format - please see below")
            print(e)
            sys.exit(1)

    def evaluate(self):

        print("Evaluating on given test data..... Please wait until next message appears.")
        # F1 score, Accuracy, Confusion Matrix, Learning Curve, Sensitivity, Specificity
        try:

            y_pred = list(self._model.predict(self._x_test))

        except Exception as e:
            print("Something went wrong!!")
            print(e)
            sys.exit(1)

        print("\n\n **************************** Classification Report on Test Data ******************************\n")
        target_names = list(map(str, list(set(list(self._y_test)))))
        report = classification_report(self._y_test, y_pred, target_names=target_names)
        print(report)
        print("\n\n***************************** F1 Score - Test Data ******************************************\n")
        f_score = f1_score(self._y_test, y_pred, average='macro')
        print(f_score)
        print("\n\n***************************** Accuracy Score - Test Data ****************************************\n")
        acc_score = accuracy_score(self._y_test, y_pred)
        print(acc_score)
        print("\n\n**************************** Confusion Matrix - Test Data **************************************\n")
        cnf_matrix = confusion_matrix(self._y_test, y_pred)
        print(cnf_matrix)
        return report, f_score, acc_score, cnf_matrix, self._model.coefs_

    def predict(self, data):

        if type(data) == 'list':
            data = np.asarray(data)
        elif type(data) == "numpy.ndarray":
            pass
        else:

            try:
                
                data = np.asarray(data)

            except Exception as e:
                print("Please make sure data to be predicted should be of type list.")
                print(e)
                sys.exit(1)

        predicted_val = list(self._model.predict(data))

        return predicted_val


if __name__ == '__main__':

    params = {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001,
              'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001,
              'power_t': 0.5, 'max_iter': 200, 'shuffle': True, 'random_state': None, 'tol': 0.0001,
              'verbose': False, 'warm_start': False, 'momentum': 0.9, 'nesterovs_momentum': True,
              'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999,
              'epsilon': 1e-08, 'n_iter_no_change': 10, 'normalization': 'min_max_norm'}
    inst = NeuralNetwork(params)
    inst.fit("mnist_data/mnist_train_data.csv", "mnist_data/mnist_test_data.csv")
    inst.evaluate()

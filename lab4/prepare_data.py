import pandas as pd


class Preparator:
    def __init__(self, data_path: str, not_negatives: list[str],
                 seperator: str, class_column: str,
                 drop_column_percentage=0.5,
                 fill_column_percentage=0.33,
                 new_dataset_path='data/cardio_proccessed.csv') -> None:
        ''' class responsible for preparing data from csv
            for classification model
        call split_dataset to get splited train/validate/test dataset from csv
            prepared for classification models

        Attributes
        :data: set of data to prepare
        :not_negatives: list of coulumns names that shouldn't be negative
        :class_column: column to predict by model
        :seperator: seperator of values in csv file
        :drop_column_percentage: percentage of wrong data to drop whole column
        :fill_column_percentage: percentage of wrong data to
            fill it with mean of the column
        :new_dataset_path: path where to save prepared dataset

        How does it work
        1. Eliminating wrong data from dataset
        - drop whole column for over drop_column_percentage percent of NaNs
        - fill them with mean for over fill_column_percentage percent of NaNs
        - drop this samples for less than fill_column_percentage percent
            of NaNs
        2. Eliminating outliers from dataset is done by deleting samples that
        are more than 4 standard devations from the mean
        '''
        try:
            self.data = pd.read_csv(data_path, sep=seperator)
        except FileNotFoundError:
            raise ValueError("Wrong path for the data in attribute data_path!")
        columns = self.data.columns.to_list()
        for column_name in not_negatives:
            if column_name not in columns or class_column not in columns:
                raise ValueError("Wrong column name!")
        self.not_negatives = not_negatives
        self.class_column = class_column
        self.drop_column_percentage = drop_column_percentage
        self.fill_column_percentage = fill_column_percentage
        self.new_dataset_path = new_dataset_path

    def _eliminate_incorrect_data(self, column_name: str,
                                  check_negatives=False):
        column_data = self.data[column_name]
        nans = negatives = 0
        dropping_value = int(self.drop_column_percentage*len(column_data))
        filling_value = int(self.fill_column_percentage*len(column_data))

        for value in column_data:
            if value is None:
                nans += 1
            if check_negatives and value < 0:
                negatives += 1

        if nans > dropping_value or negatives > dropping_value:
            self.data.drop([column_name], axis=1, inplace=True)
        elif nans > filling_value or negatives > filling_value:
            column_data.fillna(column_data.mean(), inplace=True)
        elif negatives:
            self.data = self.data[self.data[column_name] > 0]
        else:
            self.data.dropna(subset=[column_name], inplace=True)

    def _eliminate_outliers(self, column_name: str):
        mean = self.data[column_name].mean()
        std_dev = self.data[column_name].std()
        lower_bound = mean - 4 * std_dev
        upper_bound = mean + 4 * std_dev
        self.data = self.data[(self.data[column_name] >= lower_bound) &
                              (self.data[column_name] <= upper_bound)]

    def _prepare_dataset(self):
        self.data.drop_duplicates()
        for column in self.data.columns:
            check_negatives = True if column in self.not_negatives else False
            self._eliminate_incorrect_data(column, check_negatives)
            if (len(self.data[column].unique()) > 5):
                self._eliminate_outliers(column)
        self.data.to_csv(self.new_dataset_path, index=False)

    def split_dataset(self) -> tuple[pd.Series]:
        '''returns x_train, y_train, x_validate, y_validate, x_test, y_test
        divides 0.5 to 0.25 to 0.25
        '''
        self._prepare_dataset()
        training_columns = self.data.columns.to_list()
        training_columns.remove(self.class_column)
        x = self.data[training_columns].astype(float)
        y = self.data[self.class_column].astype(int)
        index = int(0.25 * len(y))

        x_test = x.iloc[:index]
        x_test.reset_index(drop=True, inplace=True)
        y_test = y.iloc[:index]
        y_test.reset_index(drop=True, inplace=True)

        x_validate = x.iloc[index:2*index]
        x_validate.reset_index(drop=True, inplace=True)
        y_validate = y.iloc[index:2*index]
        y_validate.reset_index(drop=True, inplace=True)

        x_train = x.iloc[2*index:]
        x_train.reset_index(drop=True, inplace=True)
        y_train = y.iloc[2*index:]
        y_train.reset_index(drop=True, inplace=True)

        return x_train, y_train, x_validate, y_validate, x_test, y_test

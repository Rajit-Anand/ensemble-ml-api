# preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

class DataPreprocessor:
    """
    A class for preprocessing data with methods to fit and transform datasets.
    Allows applying the same encoders across multiple datasets.
    """
    def __init__(self, numerical_columns, one_hot_columns, label_columns):
        self.numerical_columns = numerical_columns
        self.one_hot_columns = one_hot_columns
        self.label_columns = label_columns

        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        self.label_encoders = {
            col: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) for col in self.label_columns
        }
        self.one_hot_feature_names = None

    def fit(self, df):
        self.scaler.fit(df[self.numerical_columns])
        self.one_hot_encoder.fit(df[self.one_hot_columns])
        self.one_hot_feature_names = self.one_hot_encoder.get_feature_names_out(self.one_hot_columns)
        for col in self.label_columns:
            self.label_encoders[col].fit(df[[col]])

    def transform(self, df):
        df_scaled = df.copy()
        df_scaled[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        encoded_columns = self.one_hot_encoder.transform(df[self.one_hot_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=self.one_hot_feature_names, index=df.index)
        for col in self.label_columns:
            df_scaled[col] = self.label_encoders[col].transform(df[[col]])
        df_final = pd.concat([df_scaled.drop(self.one_hot_columns, axis=1), encoded_df], axis=1)
        return df_final

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

import numpy as np
import pandas as pd
import sys
from random import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import engine as eng
import compare as cp


class Descriptions:
    #This class contains general descriptions about the code and has no impact on the code whatsoever.
    # It was only created so I can write comments about the code and still collapse it whenever I want. 

    #Algorithm Design:
    # 1. Import CSV formatted data and split it into different column classes.
    # 2. Determine the type of data for each column class: High cardinality (Strings), Low cardinality (URMultCats), or scalar.
    # 3. Encode the data through one-hot encoding.
    #   a. Ignore strings (no predictive value), one hot encode URMultCats, and normalize scalars to be between 0 and 1.
    # 4. Pass encoded dataframe through WGAN engine.
    # 5. Decode the returned fake data.
    # 6. Export to file fake_cc.csv
    pass

#IMPORTANT CUSTOM VARIABLES
SVLConstant = 0.25 # "Set vs List Constant". Threshold ratio of unique values vs total values
                   # used to determine whether or not a column is type string or type category.

NumThreshold = 20 # Max possible number the category can be before being assumed to be scalar. 
                  # Can be changed later by user.

SVLNumConstant = 0.25 # Ratio of unique number values vs total number data points. Decides whether
                      # the number is scalar or representative of a category. 
                      # Used for small sample datasets.

fake_data_amt = 1000 # Number of fake data points you want to generate

batch_size = 128 # Number of data points the GAN processes before updating weights

sample_size = 3000 # Amount of data points processed in each epoch 
                    # Ex: Your dataset is 2000 data points large. If sample size is set to 300, and batch size is set to 64,
                    #     then you pull 300 random points from the dataset and then update them in groups of 64.
                    #     -The remainder of 300 / 64 is updated as a smaller batch
                    #     -Sample size cannot be larger than dataset size

epochs = 100 # Number of cycles you want your dataset to train for.

path = 'C:/Users/q1033821/Documents/VSCODE/datasets/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'

export_location = "C:/Users/q1033821/Documents/VSCODE/fake_data_factory/fake_data/fake_cc.csv"

x_value = 'Peak Direction'
y_value = 'Hmax'


#DATASETS:
#heart disease: heart-disease-uci\heart.csv
#comic book characters marvel: fivethirtyeight-comic-characters-dataset\marvel-wikia-data.csv
#comic book characters dc: fivethirtyeight-comic-characters-dataset\dc-wikia-data.csv

col_max_array = []  #List with global scope.

def main():
    real = csv_to_df(path)

    columns = categorize_df(real)
    encoded_real = encode_df(columns)

    gan = eng.GAN(encoded_real, col_max_array)

    gan.train(epochs, batch_size, sample_size)
    encoded_fake_data = gan.gen_fake_data(fake_data_amt)

    decoded_fake = decode_df(encoded_fake_data, columns)
    decoded_fake = recombine(columns)

    exported_fakes = export_data(decoded_fake, export_location)

    print(decoded_fake.head())
    print(decoded_fake.shape)

    cp.real_vs_fake(real, decoded_fake, x_value, y_value, 250)


class Category:
    #CATEGORY: The category class is a simple skeleton that is used to bind together the 
    #          different data types. This class has only 1 parameter, a string "ctype"
    #          that identifies what type of Category it is. Other categories like StrCat
    #          are also type Category, allowing for consistent usage in the Column class.
    def __init__(self):
        self.ctype = StrCat

    def set_category(self, newCategory):
        self.ctype = newCategory

    def basic_gen(self):
        #basic_gen is the default testing generator for all category types. This function is designed
        # to be overwritten by all subcategories of Category class objects.
        return 'should be overridden'

class Column:
    #COLUMN: The column class is used as the basic framework of keeping track of the
    #        different attributes of each column. When data is generated, it will refer
    #        back to the corresponding column it's in for information of its parameters.
    def __init__(self, name, realDataList):
        self.name = name
        self.cat = StrCat #The default category is string, unless other 
        self.relavant = True #User can configure later if column is irrelevant. See description for clarification.
        self.realDataList = realDataList #List of the column's corresponding real data
        self.fakeDataList = [] #List of the column's generated fake data
        self.indices = []

    def fill_col(self, numOfEntries: int):
        #Calls the column's corresponding category type's data generator function. 
        # Generates as many data points as specified by the user and puts them into the class's
        # fakeDataList parameter.
        for i in range(numOfEntries):
            self.fakeDataList.append(self.cat.basic_gen())
        return self.fakeDataList

    def identify_cat(self):
        #Identifies what category type the column is
        #HOW IT WORKS:
        #   1. Make set of unique data entries from sample data.
        #   2. Check if only 1 and 0. If so, column is binary.
        #   3. Check if set contains a string. If so move to step 3a. If not move to step 4
        #       a. Check if uniqueData set is less than total sample data * 0.25 (or whatever SVLConstant is set to)
        #       b. If so, column is multiple categories. If not, column is strings.
        #   4. Check if there exceeds 20 unique numbers (or whatever NumThreshold is set to)
        #       a. If so, column is scalar.
        #   5. Check if the set of unique numbers exceeds 0.25 * total data points (or whatever SVLNumConstant is set to)
        #       a. If so, column is scalar.
        #   6. All other options exhausted. Set column to multiple categories.
        uniqueData = set(self.realDataList)
        if(uniqueData == {1, 0}):
            self.cat = URMultCat()
            return URMultCat()
        else:
            for point in uniqueData:
                if(type(point) == str):
                    if(len(uniqueData) < SVLConstant * len(self.realDataList)):
                        self.cat = URMultCat()
                        return URMultCat()
                    else:
                        self.cat = StrCat()
                        return StrCat
            if(len(uniqueData) > NumThreshold):
                self.cat = ScaleCat()
                for point in uniqueData:
                    if point < 0:
                        self.cat.allow_negative = True
                return ScaleCat()
            elif(len(uniqueData) > SVLNumConstant * len(self.realDataList)):
                self.cat = ScaleCat()
                for point in uniqueData:
                    if point < 0:
                        self.cat.allow_negative = True
                return ScaleCat()
            self.cat = URMultCat()
            return URMultCat()

class StrCat(Category):
    lengthMin: int
    lengthMax: int

    def basic_gen(self):
        return "string"

class ScaleCat(Category):
    def __init__(self):
        self.col_max = 0
        self.scaled_data = []
        self.allow_negative = False
        return super().__init__()

    def basic_gen(self):
        return 'scalar'

class BinCat(Category):
    #DISCARD CLASS: UNNEEDED
    #ASSUME ALL BINARY CATEGORIES ARE JUST URMULT CATEGORIES. STILL FUNCTIONS THE SAME
    def basic_gen(self):
        return 'binary'

class URMultCat(Category):
    def __init__(self):
        self.col_max = 1
        self.label_encoding = LabelEncoder()
    
        return super().__init__()

    def basic_gen(self):
        return "multi-cat"

class RMultCat(Category):
    #DISCARD CLASS: CAUSES COMPLICATIONS
    #CODE REQUIRED TO IMPLEMENT RELATED CATEGORICAL DATA IS VERY DIFFICULT
    cats: list
    def basic_gen(self):
        return "rare related multi-cat"

def csv_to_df(csv_path):
        # Input: The path location of the CSV
    # Outputs: 
    #       1. The CSV scaled down to be between -1 and 1
    #       2. An array of maximum absolute values for each column. 

    real_full_data = pd.read_csv(csv_path, header=0)
    real_full_data = real_full_data.dropna()

    return real_full_data

def categorize_df(df):
    # Input: Dataframe
    # Function: Creates a list of the columns in the dataframe. 
    #   --Categorizes each column in the process.
    column_list = []
    category_list = df.columns
    for x in category_list:
        new_col = Column(x, df[x])
        new_col.identify_cat()
        column_list.append(new_col)
    return column_list

def encode_df(column_list):
    #TL;DR: One hot encodes the dataset while storing enough information to decode later on.

    encoded_df = pd.DataFrame()
    for col in column_list:
        if isinstance(col.cat, StrCat):
            continue
        

        if isinstance(col.cat, ScaleCat):
            col.indices.append(len(encoded_df.columns))
            col.cat.col_max = col.realDataList.abs().max()
            col.cat.scaled_data = col.realDataList / col.realDataList.abs().max()
            col_max_array.append(col.cat.col_max)
            encoded_df[col.name] = col.cat.scaled_data

        if isinstance(col.cat, URMultCat):
            onehot_df = pd.DataFrame()
            integer_encoded = col.cat.label_encoding.fit_transform(col.realDataList.values)
            onehot_encoder = OneHotEncoder(sparse = False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

            for i in range(len(onehot_encoded[0])):
                col.indices.append(len(encoded_df.columns))
                col_max_array.append(col.cat.col_max)

                onehot = onehot_encoded[:, i]
                encoded_df.insert(len(encoded_df.columns), col.name + str(i), onehot, allow_duplicates = True)

    return encoded_df

def decode_df(encoded_df, column_list):
    # Decodes the fake generated dataset using information from previous columns.

    for col in column_list:
        col_df = pd.DataFrame()
        i = 0
        if isinstance(col.cat, StrCat):
            col.fill_col(len(encoded_df.index))
        if isinstance(col.cat, ScaleCat):
            decoded = (encoded_df.iloc[:, col.indices[0]]).tolist() 
            col.fakeDataList.append(decoded)

        if isinstance(col.cat, URMultCat):
            for x in col.indices: 
                col_df.insert(len(col_df.columns), i, encoded_df.iloc[:, x], allow_duplicates = True)
                i = i + 1
            for ind in range(col_df[0].count()):
                decoded = col.cat.label_encoding.inverse_transform([np.argmax(col_df.iloc[ind, :])])
                col.fakeDataList.append(decoded)
                pass

    return col.fakeDataList

def recombine(column_list):
    # Reassembles each column's individual datapoints back into a full dataset.

    full_list = pd.DataFrame()
    for col in column_list:
        if isinstance(col.cat, ScaleCat):
            another_col = pd.DataFrame(col.fakeDataList).T
            another_col.columns = [col.name]
        else:
            another_col = pd.DataFrame(col.fakeDataList) 
            another_col.columns = [col.name]
        full_list = pd.concat([full_list, another_col], axis = 1)

    return full_list


def export_data(data, destination):
    export_csv = data.to_csv(destination)
    return export_csv

main()
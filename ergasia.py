import operator
import sqlite3
from sqlite3.dbapi2 import Connection
import numpy as np
import pandas as pd
from scipy.sparse.extract import find
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

# establish connection with the database and execute query to get tables
connection = sqlite3.connect("database.sqlite")
cur = connection.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")   

# print("TABLES: ")
# print(cur.fetchall())

# load only the tables we will use and convert them into Pandas DataFrames
match = pd.read_sql_query("SELECT * FROM Match", connection)
team_attributes = pd.read_sql_query("SELECT * FROM Team_Attributes", connection)

match['date'] = pd.to_datetime(match['date'])
team_attributes['date'] = pd.to_datetime(team_attributes['date'])
match['year'] = match['date'].dt.year
team_attributes['year'] = team_attributes['date'].dt.year

# keep only the columns that we'll need during the training phase
match_keep = [
    'year',
    'home_team_api_id', 'away_team_api_id', 
    'home_team_goal', 'away_team_goal', 
    'B365H', 'B365D', 'B365A',
    'BWH', 'BWD', 'BWA', 
    'IWH', 'IWD', 'IWA', 
    'LBH', 'LBD', 'LBA'
    ]
match = match[match_keep]
# remove all the records whose prediction vectors have value = 0
match = match.replace(np.nan, 0.0)
match = match.dropna()

team_attributes_keep = [
    'year', 'team_api_id', 
    'buildUpPlaySpeed','buildUpPlayPassing', 
    'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 
    'defencePressure', 'defenceAggression','defenceTeamWidth'
    ]
team_attributes = team_attributes[team_attributes_keep]
# remove all the records whose prediction vectors have value = 0
team_attributes = team_attributes.replace(np.nan, 0.0)
team_attributes = team_attributes.dropna()

# print("\n---MATCH---\n")
# print(match)
# print("\n---TEAM_ATTRIBUTES---   \n")
# print(team_attributes)

# create training vectors
def train(matches, teams):
    t_set = []
    for i, row in matches.iterrows():
        home_team_id = row['home_team_api_id']
        away_team_id = row['away_team_api_id']

        home_team_data = teams[(teams.team_api_id == home_team_id)]
        home_team_data = home_team_data.drop(columns = 'team_api_id')
        home_team_data.columns = 'HOME' + home_team_data.columns
        away_team_data = teams[(teams.team_api_id == away_team_id)]
        away_team_data = away_team_data.drop(columns = 'team_api_id')
        away_team_data.columns = 'AWAY' + away_team_data.columns

        try:
            join = pd.concat([row, home_team_data.iloc[1], away_team_data.iloc[1]], axis = 0, join = "inner")
        except:
            continue            
        t_set.append(join)

    return t_set

# create training set using the above function
train_x = train(match, team_attributes)
train_x = pd.DataFrame(train_x)
train_x = train_x.drop(columns = ['year', 'home_team_api_id', 'away_team_api_id', 'AWAYyear', 'HOMEyear'])
# find the result of the game and store it in a seperate column
train_x['r'] = train_x['home_team_goal'] - train_x['away_team_goal']    

# print(train_x.columns)

# create a dictionary for training based on the betting company 
companies = {
    'B364': ['B365H', 'B365D', 'B365A' ,'r'],
    'BW':   ['BWH', 'BWD', 'BWA', 'r'],
    'LW':   ['IWH', 'IWD', 'IWA', 'r'],
    'LB':   ['LBH', 'LBD', 'LBA', 'r']
}

# scale the data
scaler = MinMaxScaler()
kfold = KFold(n_splits = 10)

# find the betting company with the best predictions 
def find_best(accuracies):
    bookers = ['B364', 'BW', 'LW', 'LB']
    i, value = max(enumerate(accuracies), key = operator.itemgetter(1))

    return bookers[i]

# find the result of the game 
def find_result(data):
    result = [] 
    for i, j in enumerate(data):
        if j > 0:
            result.append('H')
        elif j == 0:
            result.append('D')
        else:
            result.append('A')

    return result

# # QUESTION 1
# acc = []
# for cmp in companies:
#     train = train_x[companies[cmp]].reset_index(drop = True)
#     y = train['r']
#     x = scaler.fit_transform(train.drop(columns = ['r']))

#     print(f'BETTING COMPANY: {cmp}')    
#     kfold.get_n_splits(x)
#     k = 1
#     accuracy = 0
#     for train_i, test_i in kfold.split(x):
#         x_train, x_test = x[train_i], x[test_i]
#         y_train, y_test = y[train_i], y[test_i]
#         y_train = np.array(y_train)

#         model = Sequential()
#         model.add(Dense(1, activation = "relu"))
#         model.compile(
#             loss = 'mean_squared_error', 
#             optimizer = 'adam', 
#             metrics = ['accuracy']
#             )
#         model.fit(x_train, y_train, epochs = 3)
        
#         prd = scaler.transform(x_test)
#         prd = model.predict(prd)
#         prd = find_result(prd)
#         y_test = find_result(y_test)
#         print(f'Fold Number: {k}, {accuracy_score(y_test, prd)}')
#         accuracy = accuracy + accuracy_score(y_test, prd)
#         k = k + 1                      
#     k = 0
#     accuracy = accuracy / 10
#     acc.append(accuracy)
    
# print(f'The best betting company is: {find_best(acc)}')

# # QUESTION 2
# regression = linear_model.LinearRegression()
# acc = []
# for cmp in companies:
#     train = train_x[companies[cmp]].reset_index(drop = True)
#     y = train['r']
#     x = scaler.fit_transform(train.drop(columns = ['r']))

#     print("BETTING COMPANY: " + cmp)    
#     kfold.get_n_splits(x)
#     k = 1
#     accuracy = 0
#     for train_index, test_index in kfold.split(x):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         regression.fit(x_train, y_train)
        
#         prd = scaler.transform(x_test)
#         prd = regression.predict(prd)
#         prd = find_result(prd)
#         y_test = find_result(y_test)
#         print(f'Fold Number: {k}, {accuracy_score(y_test, prd)}')
#         accuracy = accuracy + accuracy_score(y_test, prd)
#         k = k + 1                      
#     k = 0
#     accuracy = accuracy / 10
#     acc.append(accuracy)
    
# print(f'The best betting company is: {find_best(acc)}')

# # QUESTION 3
# y = train_x['r'] 
# x = scaler.fit_transform(train_x.drop(columns = ['home_team_goal', 'away_team_goal', 'r']))     # drop columns we no longer need 

# kfold.get_n_splits(x)
# k = 1
# accuracy = 0
# for train_index, test_index in kfold.split(x):
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     y_train = np.array(y_train)
    
#     model = Sequential()
#     model.add(Dense(14, activation = "sigmoid"))
#     model.add(Dense(14, activation = "sigmoid"))
#     model.add(Dense(1, activation = "sigmoid")) 
#     model.compile(
#         loss = 'mean_squared_error', 
#         optimizer = 'adam', 
#         metrics = ['accuracy']
#         )
#     model.fit(x_train, y_train, epochs = 3)
        
#     prd = scaler.transform(x_test)
#     prd = model.predict(prd)
#     prd = find_result(prd)
#     y_test = find_result(y_test)
#     print(f'Fold Number: {k}, {accuracy_score(y_test, prd)}')
#     accuracy = accuracy + accuracy_score(y_test, prd)
#     k += 1

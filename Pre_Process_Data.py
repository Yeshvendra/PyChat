#######################################################################################################
# FILE DESCRIPTION
# This file contains code for data pre processing which will finally be used in the PyChat project.
#######################################################################################################

import json
import pandas as pd

# Reading python code data and droping index column
df = pd.read_csv('./Dataset/ProblemSolutionPythonV3.csv')
df = df.drop(columns=["Unnamed: 0"])
df = df.reset_index()

# Changing column name for convenience
df = df.rename(columns={"Problem": 'text', 'Python Code': 'code'})

# Opening out of scope data JSON file
fallBackJSONFile = open('./Dataset/OutOfScopeData.json')
  
# returns JSON object as for out of scope data as a dictionary
fallBackDict = json.load(fallBackJSONFile)

final_dataset_dict = {'query':[], 'response': []}

outOfScopeIntentDict = {}

statementCount = 0
# Iterating through the json list to create a dictionary of statemens with intent as key
for key in fallBackDict:
    for fallBackItem in fallBackDict[key]:
        statementCount += 1
        if fallBackItem[1] in outOfScopeIntentDict:
           outOfScopeIntentDict[fallBackItem[1]].append(fallBackItem[0])
        else:
           outOfScopeIntentDict[fallBackItem[1]] = [fallBackItem[0]]

print("Number of different intents: ", len(outOfScopeIntentDict))
print("Number of out-of-scope statements: ", statementCount)
    
# Adding Out-Of-Scope intent to the final dataset dictionary
for key in outOfScopeIntentDict:
    textCount = 0
    for text in outOfScopeIntentDict[key]:
        final_dataset_dict['query'].append(text)
        if key == "greeting":
            final_dataset_dict['response'].append('Hello! I am PyChat your python code Question & Answer chatbot. You can ask me to write a python program. Go ahead give it a shot.')
        else:
            final_dataset_dict['response'].append('PyChat is only meant to answer Python Code Questions and your question seems a little out of scope. Would you like to try again with a new question?')
        textCount += 1
        if textCount > 22 and key != "greeting":
            break

# Adding python code to the final dataset
for index, row in df.iterrows():
    code = row['code']
    final_dataset_dict['query'].append(row['text'].strip())
    final_dataset_dict['response'].append("\"" + str(code) + "\"")

final_df = pd.DataFrame(final_dataset_dict)
print(final_df.head())
final_df.to_csv('./Dataset/FinalDataset.csv')

print("Creating the dataset file to be used in the project is complete!!")

# Closing file
fallBackJSONFile.close()

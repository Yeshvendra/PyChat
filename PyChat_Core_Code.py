#######################################################################################################
# FILE DESCRIPTION
# This file contains core logic of the Pychat where we use ALBERT to do word embedding followed by 
# finding cosine similarity to generate an appropriate response.
#######################################################################################################

import sister
import numpy as np
import pandas as pd
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

#############SECTION FOR GLOBAL VARIABLES##############
CSV_FILE_NAME = './Dataset/FinalDataset.csv'
# Initialize the ALBERT word embedding module
EMBEDDER = sister.MeanEmbedding(lang="en")
LOG_CONFIDENCE = 0
projectDataframe = pd.DataFrame()
#######################################################

# Main function which would run when this file is directly run without the server code
def main():
    print('\n\n***************PyChat***************\n')
    print('Welcome to the PyChat Console.')
    print('\n***************PyChat***************\n\n')
    while True:
        print('===============================================================================\n')
        query = input('Input Query:\n')

        response, confident  = GenerateResponse(query)

        if LOG_CONFIDENCE:
            print(f"Confident: {confident}%")
        PrintResponse(response)
        print('\n===============================================================================\n\n')

# Function to read dataset which was already created
def ReadDataset():
    global projectDataframe
    # Read csv file and create a dataframe
    projectDataframe = pd.read_csv(CSV_FILE_NAME)

# Function to preprocess the dataset
def PreProcessDataset():
    global projectDataframe
    # Remove the first column as it only contains index
    projectDataframe = projectDataframe.drop(columns=["Unnamed: 0"])
    
    # Remove any row which has NA in it
    projectDataframe = projectDataframe.dropna()

    # Preprocess text from data for later use
    projectDataframe['preprocessed_query'] = projectDataframe['query'].apply(lambda line: RemoveStopWords(line))
    cols = ['query', 'preprocessed_query', 'response']
    projectDataframe = projectDataframe[cols]

    CreateWordEmbedding()

# Function to create word embedding using Albert and store it in dataframe
def CreateWordEmbedding():
    global projectDataframe

    projectDataframe['embedded_query'] = projectDataframe['preprocessed_query'].apply(
                                            lambda line: EMBEDDER(line) if line != "" else np.array([]))

# Function to remove stopwords and convert text to lower case
def RemoveStopWords(sent):
    return " ".join([word.lower() for word in tokenize(str(sent)) if word not in stopwords.words('english')])

# Function to find cosine similarity of the two datapoints
def FindCosineSimilarity(embeddedInputQuery, embeddedDatasetQuery):
    cosineSimilarity = 1
    if embeddedDatasetQuery.size == 0:
        cosineSimilarity = 1
    else:
        cosineSimilarity = cosine(embeddedInputQuery, embeddedDatasetQuery)
    return cosineSimilarity

# Function to find appropriate response and return it back
def GenerateResponse(inputQuery):
    global projectDataframe
    preprocessedInputQuery = RemoveStopWords(inputQuery)
    embeddedInputQuery = EMBEDDER(preprocessedInputQuery)
    results = [FindCosineSimilarity(embeddedInputQuery, embeddedDatasetQuery) 
                for embeddedDatasetQuery in projectDataframe['embedded_query']]
    similarity = min(results)
    return projectDataframe['response'][results.index(similarity)], round((1-similarity)*100, 2)

# Format the code before printing it
def PrintResponse(response):
    # Removing first " as we dont want to show code between quotes
    if response[0] == "\"":
        response = response[1:]
    # Removing last " as we dont want to show code between quotes
    if response[-1] == "\"":
        response = response[:-1]
    print("\nThe reponse:\n--------------\n")
    for line in response.split('\r'):
        print(line)


ReadDataset()
PreProcessDataset()
print("Loading & Preprocessing of data complete...")

if __name__=="__main__":
    main()

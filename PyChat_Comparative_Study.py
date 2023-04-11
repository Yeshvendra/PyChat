#######################################################################################################
# FILE DESCRIPTION
# This file contains comparative study of the Pychat where we use ALBERT and TFIDF text vectorization 
# technique and compare them to find our state of the art.
#######################################################################################################

import sister
import pandas as pd
import numpy as np
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

#############SECTION FOR GLOBAL VARIABLES##############
CSV_FILE_NAME = './Dataset/FinalDataset.csv'
# Initialize the ALBERT word embedding module
ALBERT_EMBEDDER = sister.MeanEmbedding(lang="en")
VECTORIZER = CountVectorizer(ngram_range=(1,1))
TFIDF = TfidfTransformer()
SVD = TruncatedSVD(n_components=400, n_iter=2, random_state=42)
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

        response_albert, confident_albert  = GenerateResponse_Albert(query)
        response_tfidf, confident_tfidf = GenerateResponse_Tfidf(query)

        print("\n\n++++++++++++++++++++++++Response of Albert Embedding++++++++++++++++++++++++++++++\n\n")
        if LOG_CONFIDENCE:
            print(f"Confident: {confident_albert}%")
        PrintResponse(response_albert)

        print("\n\n++++++++++++++++++++++++Response of TF-IDF Embedding++++++++++++++++++++++++++++++\n\n")
        if LOG_CONFIDENCE:
            print(f"Confident: {confident_tfidf}%")
        PrintResponse(response_tfidf)
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
    CreateTFIDFVector()

# Function to text vectors using TFIDF and store it in dataframe
def CreateTFIDFVector():
    global projectDataframe
    # Create Bag of Words
    bowTransform = VECTORIZER.fit_transform(projectDataframe['preprocessed_query'])
    # Apply TFIDF
    tfidfTransform = TFIDF.fit_transform(bowTransform)
    # Apply SVD to deal with the sparsity 
    svdTransform = SVD.fit_transform(tfidfTransform)
    svdDataframeList = []
    for row in svdTransform:
        svdDataframeList.append(row)
    projectDataframe['tfidfvectorized_query'] = svdDataframeList

# Function to create word embedding using Albert and store it in dataframe
def CreateWordEmbedding():
    global projectDataframe

    projectDataframe['embedded_query'] = projectDataframe['preprocessed_query'].apply(
                                            lambda line: ALBERT_EMBEDDER(line) if line != "" else np.array([]))

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

# Function to calculate similarity based on albert embedding features
# to find appropriate response and return it back
def GenerateResponse_Albert(inputQuery):
    global projectDataframe

    preprocessedInputQuery = RemoveStopWords(inputQuery)

    embeddedInputQuery = ALBERT_EMBEDDER(preprocessedInputQuery)
    
    results = [FindCosineSimilarity(embeddedInputQuery, embeddedDatasetQuery) 
                for embeddedDatasetQuery in projectDataframe['embedded_query']]
    similarity = min(results)
    return projectDataframe['response'][results.index(similarity)], round((1-similarity)*100, 2)

# Function to calculate similarity based on tfidf features
# to find appropriate response and return it back
def GenerateResponse_Tfidf(inputQuery):
    global projectDataframe

    preprocessedInputQuery = RemoveStopWords(inputQuery)

    bowTransform = VECTORIZER.transform([preprocessedInputQuery])
    tfidfTransform = TFIDF.transform(bowTransform)
    svdTransform = SVD.transform(tfidfTransform)

    embeddedInputQuery = svdTransform[0]
    
    results = [FindCosineSimilarity(embeddedInputQuery, embeddedDatasetQuery) 
                for embeddedDatasetQuery in projectDataframe['tfidfvectorized_query']]
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

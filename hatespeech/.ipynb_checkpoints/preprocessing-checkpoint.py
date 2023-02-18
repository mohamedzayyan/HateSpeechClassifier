import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/input/code/my_package/requirements.txt",
])

#!pip install nltk
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
nltk.download([
     "names",
     "stopwords",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
        "wordnet",
        'omw-1.4'
])

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/labeled-data.csv"
    )
    textData = df.tweet.apply(lambda x: x.lower()).tolist()
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = WordNetLemmatizer()
    documents = []
    for sen in range(0, len(textData)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(textData[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        #Removing "rt" tag
        document = re.sub('rt ', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords)
    X_pre = vectorizer.fit_transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    X_pre = tfidfconverter.fit_transform(X_pre).toarray()
    y_pre = df['class'].values.reshape((-1,1))

    
    X = np.concatenate((y_pre, X_pre), axis=1)
    
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])

    
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
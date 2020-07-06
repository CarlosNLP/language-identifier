import sqlite3
import re
import xml.etree.ElementTree as ET
import string
import numpy as np
from nltk.tokenize import word_tokenize

# Method to parse a TMX file and dump its contents into a database
def tmx_to_sql(db_path, tmx_path):
    # Opening and connecting to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Creating table if not exists
    cur.execute("""CREATE TABLE IF NOT EXISTS "translations" (
                "source_text"    TEXT,
                "target_text"    TEXT,
                "source_lang"   TEXT,
                "target_lang"   TEXT
                );""")
    
    # Running some replacements in TMX file
    try:
        with open(tmx_path, 'rt', encoding="utf-8-sig") as f:
            content = f.read()
            content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
            content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
            content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
        with open(tmx_path, 'wt', encoding="utf-8-sig") as f:
            f.write(content)
    except: # trying reading with UTF-16 encoding and writing as UTF-8 BOM so the ET parser recognizes as XML
        try:
            with open(tmx_path, 'rt', encoding='utf16') as f:
                content = f.read()
                content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
                content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
                content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
            with open(tmx_path, 'wt', encoding="utf-8-sig") as f:
                f.write(content)
        except: # showing file with error
            print('Error reading file', tmx_path)
    
    # Parsing TMX file
    doc = ET.parse(tmx_path)
    root = doc.getroot()
    tu_list = root.findall("./body/tu")
    
    # Looping through every translation unit
    for tu in tu_list:
        # Getting the translation unit pairs
        tuv = tu.findall("tuv")
        
        # Retrieving specific values for our database
        source_text = tuv[0].find("seg").text
        target_text = tuv[1].find("seg").text
        source_lang = tuv[0].get("lang").lower()
        target_lang = tuv[1].get("lang").lower()
        
        # Inserting entries into our database
        cur.execute("INSERT INTO translations VALUES (?, ?, ?, ?)", (source_text, target_text, source_lang, target_lang))
    
    # Committing the changes into the database
    conn.commit()
    
    # Closing database connection
    cur.close()
    conn.close()


# Method to clean or pre-process the sentence before its use
def process_sentence(sentence):
    '''
    Input:
        sentence: a string containing the retrieved sentence
    Output:
        sentence_clean: a list of words containing the processed sentence
    '''
    # Removing punctuation from the sentence
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Tokenizing sentence
    sentence_tokens = word_tokenize(sentence)
    
    sentence_clean = []
    for word in sentence_tokens:
        sentence_clean.append(word.lower()) # appending lowered word
    
    return sentence_clean


# Method to build the frequency dictionary (word, label) > frequency
def count_sentences(result, sentences, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        sentences: a list of sentences
        ys: a list corresponding to the label of each sentence (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    
    # Looping through the sentences and ys
    for sentence, y in zip(sentences, ys):
        for word in process_sentence(sentence): # looping through every word from the sentence
            # Defining the key of the dictionary, which is a tuple composed of word and label
            pair = (word, y)
            
            # Incrementing the count if the pair already exists in the dictionary
            if pair in result:
                result[pair] += 1
            
            # Adding 1 to the dictionary if the pair is new for the dictionary
            else:
                result[pair] = 1
    
    return result


# Method mapping word and label with its frequency
def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (word, label)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears
    '''
    n = 0 # if the pair is not found, it will output 0 for its frequency
    
    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]
    
    return n


# Method to train the model
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of sentences
        train_y: a list of labels correponding to the sentences (0,1)
    Output:
        logprior: the log prior
        loglikelihood: the log likelihood of the Naive bayes equation
    '''
    loglikelihood = {}
    logprior = 0
    
    # Calculate V, which is the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    # Calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = V_pos = V_neg = 0
    for pair in freqs.keys():
        # if the label is 1
        if pair[1] == 1:
            # Incrementing the count of unique positive words by 1
            V_pos += 1
            
            # Incrementing the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]
            
        # else, the label is negative
        else:
            # Incrementing the count of unique negative words by 1
            V_neg += 1
            
            # Incrementing the number of negative words by the count for this (word, label) pair
            N_neg += freqs[pair]

    # Calculating D, which is the number of sentences
    D = len(train_y)
    
    # Calculating D_pos, which is the probability of a sentence being positive
    D_pos = len([label for label in train_y if label == 1]) / D
    
    # Calculating D_neg, which is the probability of a sentence being negative
    D_neg = len([label for label in train_y if label == 0]) / D
    
    # Calculating logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    # Looping through every word in the vocabulary
    for word in vocab:
        # Getting the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)
        
        # Calculating the probability of the word being is positive and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        
        # Calculating the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    
    return logprior, loglikelihood


# Method to predict the loglikelihood of a sentence
def naive_bayes_predict(sentence, logprior, loglikelihood):
    '''
    Input:
        sentence: a string containing the specific sentence
        logprior: a number containing your learned logprior
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the sentence (if found in the dictionary) + logprior (a number)
    '''
    # Processing the sentence to get a list of words
    word_l = process_sentence(sentence)
    
    # Initializing probability to zero
    p = 0
    
    # Add the logprior
    p += logprior
    
    # Looping through every word from the list
    for word in word_l:
        # Checking if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # Add the log likelihood of that word to the probability
            p += loglikelihood[word]
    
    return p


# Method to get the accuracy of the model with a given testing set
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: a list of sentences
        test_y: the corresponding labels for the list of sentences
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of sentences classified correctly)/(total # of tweets)
    """
    accuracy = 0  # initializing the accuracy to 0
    
    y_hats = []
    for sentence in test_x: # looping through the sentences from the testing set
        # if the prediction is > 0
        if naive_bayes_predict(sentence, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0
        
        # Appending the predicted class to the list y_hats
        y_hats.append(y_hat_i)
    
    # Error is the average of the absolute values of the differences between y_hats and test_y
    wrong_preds = 0
    for i in range(len(y_hats)):
        if y_hats[i] != test_y[i]:
            wrong_preds += 1
    error = wrong_preds / len(y_hats)
    
    # Accuracy is 1 minus the error
    accuracy = 1 - error
    
    return accuracy

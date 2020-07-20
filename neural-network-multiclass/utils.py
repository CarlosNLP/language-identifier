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
            print('Reading', tmx_path)
            content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
            content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
            content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
        with open(tmx_path, 'wt', encoding="utf-8-sig") as f:
            f.write(content)
    except: # trying reading with UTF-16 encoding and writing as UTF-8 BOM so the ET parser recognizes as XML
        try:
            with open(tmx_path, 'rt', encoding='utf16') as f:
                content = f.read()
                print('Reading', tmx_path)
                content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
                content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
                content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
                content = content.replace("UTF-16LE", "utf-8") # avoiding encoding issue when parsing TMX
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

        for i in range(1, len(tuv)):
            # Retrieving specific values for our database
            source_text = tuv[0].find("seg").text
            target_text = tuv[i].find("seg").text
            source_lang = tuv[0].get("lang").lower()
            target_lang = tuv[i].get("lang").lower()
            
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
    sentence = re.sub('[0-9]', '', sentence) # removing numbers
    # Tokenizing sentence
    sentence_tokens = word_tokenize(sentence)
    
    sentence_clean = []
    for word in sentence_tokens:
        sentence_clean.append(word) # not lowering the words have a better performance (i.e: the Italian 'i' gives much more frequencies than English 'I')
    
    return sentence_clean


# Method to build the frequency dictionary (word, label) > frequency
def build_freqs(result, sentences, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        sentences: a list of sentences
        ys: a list corresponding to the label of each sentence (from 0 to 4)
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


def extract_features(sentence, freqs):
    '''
    Input: 
        sentence: a sentence from train_x
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,6)
    '''
    # process_sentence tokenizes and removes punctuation
    word_l = process_sentence(sentence)
    
    # 6 elements in the form of a 1 x 6 vector
    x = np.zeros((1, 6)) 
    
    # bias term is set to 1
    x[0,0] = 1
    
    # loop through each word in the list of words
    for word in word_l:
        # increment the word frequency for the label 0 (English)
        x[0,1] += freqs.get((word, 0), 0)
        
        # increment the word frequency for the label 1 (French)
        x[0,2] += freqs.get((word, 1), 0)
        
        # increment the word frequency for the label 2 (Italian)
        x[0,3] += freqs.get((word, 2), 0)
        
        # increment the word frequency for the label 3 (German)
        x[0,4] += freqs.get((word, 3), 0)
        
        # increment the word frequency for the label 4 (Spanish)
        x[0,5] += freqs.get((word, 4), 0)
    
    return x

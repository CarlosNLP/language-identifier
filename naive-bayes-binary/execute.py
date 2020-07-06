import numpy as np
import os
from utils import *

# Defining the path to the DB and TMX folder
db_path = "dataset/translations.db"
tmx_folder = "dataset"

# Set to True if you would like to convert the TMX to the database, or False otherwise
convert_tmx_to_sql = True

if convert_tmx_to_sql == True:
    print('Converting TMX to SQL: True')
    
    # Creating a list with all the TMX files
    tmx_list = []
    for root, dirs, files in os.walk(tmx_folder):
        for name in files:
            if name.endswith('.tmx'):
                tmx_list.append(os.path.join(root, name))
    
    # Dumping TMX data into SQL database
    print('Number of TMX files:', len(tmx_list))
    for tmx_path in tmx_list:
        tmx_to_sql(db_path, tmx_path)

else: # printing for reference
    print('Converting TMX to SQL: False')

    
# Path and opening database
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Getting a list of English and Spanish sentences from the SQL database
cur.execute("SELECT * FROM translations WHERE source_lang = 'en-gb' AND target_lang = 'es-es'")
data = cur.fetchall()

# Initializing English and Spanish sentences lists
english_sentences = []
spanish_sentences = []

# Looping through the translation units from the fetched data
for entry in data:
    english_sentences.append(entry[0])
    spanish_sentences.append(entry[1])

print('\nEnglish sentences in database:', len(english_sentences))
print('Spanish sentences in database:', len(spanish_sentences))
print('English label: 1', 'Spanish label: 0')

# Splitting data into training and testing sets
# The values should be modified depending on the number of data you have, in my case more than 51,000 sentences per language in total
train_english = english_sentences[:50000]
test_english = english_sentences[50000:]
train_spanish = spanish_sentences[:50000]
test_spanish = spanish_sentences[50000:]

# Defining x: English/Spanish sentences
train_x = train_english + train_spanish
test_x = test_english + test_spanish

# Defining y: English 1, Spanish 0
train_y = np.append(np.ones(len(train_english)), np.zeros(len(train_spanish)))
test_y = np.append(np.ones(len(test_english)), np.zeros(len(test_spanish)))

# Confirming that our data is correct so far
print('\ntrain_english:', len(train_english))
print('test_english:', len(test_english))
print('train_spanish:', len(train_spanish))
print('test_spanish:', len(test_spanish))
print('train_x:', len(train_x))
print('test_x:', len(test_x))
print('train_y:', len(train_y))
print('test_y:', len(test_y))

# Builing the freqs dictionary
freqs = count_sentences({}, train_x, train_y)

# Training the model
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

# Calculating model accuracy
print("\nModel accuracy: %0.4f" % (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

# Trying to predict with my own sentence
my_sentence = '¡Aprender sobre NLP está muy interesante!'
print('\nMy sentence:', my_sentence)

# Getting prediction
p = naive_bayes_predict(my_sentence, logprior, loglikelihood)
print(p)

if p > 0:
    print('That\'s English!')
else:
    print('¡Eso es español!')


# Closing database connection
cur.close()
conn.close()

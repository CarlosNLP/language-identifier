import numpy as np
import os
from utils import *
import tensorflow as tf

# Defining the path to the DB and TMX folder
db_path = "dataset/translations.db"
tmx_folder = "dataset"

# Set to True if you would like to convert the TMX to the database, or False otherwise
convert_tmx_to_sql = False

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

# Initializing English and FIGS sentences lists
english_sentences = []
french_sentences = []
italian_sentences = []
german_sentences = []
spanish_sentences = []

# Getting a list of English sentences from the SQL database
cur.execute("SELECT source_text FROM translations")
data = cur.fetchall()

# Updating the English sentences list
for entry in data:
    english_sentences.append(entry[0])

# Getting a list of FIGS sentences from the SQL database
cur.execute("SELECT target_text, target_lang FROM translations WHERE target_lang = 'fr-fr' OR target_lang = 'it-it' OR target_lang = 'de-de' OR target_lang = 'es-es'")
data = cur.fetchall()

# Updating the FIGS sentences list
for entry in data:
    if entry[1] == 'fr-fr':
        french_sentences.append(entry[0])
    elif entry[1] == 'it-it':
        italian_sentences.append(entry[0])
    elif entry[1] == 'de-de':
        german_sentences.append(entry[0])
    elif entry[1] == 'es-es':
        spanish_sentences.append(entry[0])

# Closing database connection
cur.close()
conn.close()

# Keeping just the unique values from the lists
english_sentences = list(set(english_sentences))
french_sentences = list(set(french_sentences))
italian_sentences = list(set(italian_sentences))
german_sentences = list(set(german_sentences))
spanish_sentences = list(set(spanish_sentences))

print('\nEnglish sentences in database:', len(english_sentences))
print('French sentences in database:', len(french_sentences))
print('Italian sentences in database:', len(italian_sentences))
print('German sentences in database:', len(german_sentences))
print('Spanish sentences in database:', len(spanish_sentences))

# Splitting data into training and testing sets
# The values should be modified depending on the number of data you have, I am just keeping 100,000 sentences per language in total
print('\nRetrieving 100,000 entries per language...')
train_english = english_sentences[:99500]
test_english = english_sentences[99500:100000]

train_french = french_sentences[:99500]
test_french = french_sentences[99500:100000]

train_italian = italian_sentences[:99500]
test_italian = italian_sentences[99500:100000]

train_german = german_sentences[:99500]
test_german = german_sentences[99500:100000]

train_spanish = spanish_sentences[:99500]
test_spanish = spanish_sentences[99500:100000]

# Defining x: English + FIGS for both training and testing sets
train_x = train_english + train_french + train_italian + train_german + train_spanish
test_x = test_english + test_french + test_italian + test_german + test_spanish

# Initializing y for training set
train_y_english = np.zeros(len(train_english)) # [0 0 0 0 0 ... 0 0 0 0 0]
train_y_french = np.ones(len(train_french)) # [1 1 1 1 1 ... 1 1 1 1 1]
train_y_italian = np.full(len(train_italian), 2) # [2 2 2 2 2 ... 2 2 2 2 2]
train_y_german = np.full(len(train_german), 3) # [3 3 3 3 3 ... 3 3 3 3 3]
train_y_spanish = np.full(len(train_spanish), 4) # [4 4 4 4 4 ... 4 4 4 4 4]

train_y = np.concatenate((train_y_english, train_y_french, train_y_italian, train_y_german, train_y_spanish))

# Initializing y for testing set
test_y_english = np.zeros(len(test_english)) # [0 0 0 0 0 ... 0 0 0 0 0]
test_y_french = np.ones(len(test_french)) # [1 1 1 1 1 ... 1 1 1 1 1]
test_y_italian = np.full(len(test_italian), 2) # [2 2 2 2 2 ... 2 2 2 2 2]
test_y_german = np.full(len(test_german), 3) # [3 3 3 3 3 ... 3 3 3 3 3]
test_y_spanish = np.full(len(test_spanish), 4) # [4 4 4 4 4 ... 4 4 4 4 4]

test_y = np.concatenate((test_y_english, test_y_french, test_y_italian, test_y_german, test_y_spanish))

# Confirming that our data is correct so far
print('\ntrain_english:', len(train_english))
print('train_french:', len(train_french))
print('train_italian:', len(train_italian))
print('train_german:', len(train_german))
print('train_spanish:', len(train_spanish))

print('\ntest_english:', len(test_english))
print('test_french:', len(test_french))
print('test_italian:', len(test_italian))
print('test_german:', len(test_german))
print('test_spanish:', len(test_spanish))

print('\ntrain_x:', len(train_x))
print('test_x:', len(test_x))

print('\ntrain_y:', len(train_y))
print('test_y:', len(test_y))

# Builing the freqs dictionary
print('\nBuilding the freqs dictionary...')
freqs = build_freqs({}, train_x, train_y)

# Collecting the features 'x' and stack them into a matrix 'X'
print('Collecting the features for the training set...')
X_train = np.zeros((len(train_x), 6))
for i in range(len(train_x)):
    X_train[i, :] = extract_features(train_x[i], freqs)

# Training labels corresponding to X
Y_train = train_y

print('\nShape of X_train:', X_train.shape)
print('Shape of Y_train:', Y_train.shape)

# Normalizing training data
print('\nNormalizing training data...')
tf.keras.utils.normalize(X_train, axis=1)

# Defining the model
print('\nDefining model architecture...')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

print('Compiling the model...')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

print('Training the model...')
model.fit(X_train, Y_train, epochs=3)

# Calculating accuracy
print('\nCollecting the features for the testing set...')
X_test = np.zeros((len(test_x), 6))
for i in range(len(test_x)):
    X_test[i, :] = extract_features(test_x[i], freqs)

Y_test = test_y

print('\nShape of X_test:', X_test.shape)
print('Shape of Y_test:', Y_test.shape)

# Normalizing testing data
print('\nNormalizing testing data...')
tf.keras.utils.normalize(X_test, axis=1)

print('\nEvaluating the model...')
val_loss, val_acc = model.evaluate(X_test, Y_test)
print(val_loss, val_acc)

# Testing with my own data
predictions_path = "predictions"
predictions_files = []

for root, dirs, files in os.walk(predictions_path):
    for name in files:
        predictions_files.append(os.path.join(root, name))

for file in predictions_files:
    with open(file, 'rt', encoding="utf-8") as f:
        content = f.read()
        X_own = extract_features(content, freqs)
        prediction_own = model.predict(X_own)
        if np.argmax(prediction_own) == 0:
            lang = "English"
        elif np.argmax(prediction_own) == 1:
            lang = "French"
        elif np.argmax(prediction_own) == 2:
            lang = "Italian"
        elif np.argmax(prediction_own) == 3:
            lang = "German"
        elif np.argmax(prediction_own) == 4:
            lang = "Spanish"
        
        print(file, lang)

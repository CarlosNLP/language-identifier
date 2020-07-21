Naive Bayes binary model

The purpose of this model is to predict if a sentence given by the user is written in English or Spanish (binary classification).

Python files:
utils.py >> helper file to get the methods called from execute.py
execute.py >> main file that will train the model, get the accuracy with the testing set and predict your given sentence (line 87)

NOTE: if you haven't built the SQL database yet (it's done automatically by running the execute.py file), you'll need to place a TMX file, or a set of TMX files, into a folder "dataset" in the same directory and set the variable convert_tmx_to_sql to True (line 10 of execute.py). If you have already built the SQL database, you can skip this task (in order not to reimport into the database the same stuff) by setting the variable convert_tmx_to_sql to False. But take into account that lines 54-57 from execute.py depend on the size of your data. More information below.

The "dataset" directory should include either a TMX file or a database. For this task, I used a 32MB TMX file (more than 51,000 translation units) and it resulted in a 16MB database, but you can use the resources you would like. I am leaving this folder empty because GitHub has a limit for file uploads, but the TMX used for this test (99.1% accuracy) can be found here: https://www.ttmem.com/terminology/download-translation-memory/european-commission-translation-memory/. Filter by 'English' as source and 'Spanish' as target and download the one from European Commission terminology (DGT) with 51,107 translation units (EN_ES_EUR.zip). After downloading it, place it under the "dataset" directory, make sure the convert_tmx_to_sql variable is set to True (line 10 of execute.py) and run the file. For the next usages, if you are not importing anything else and you just want to try out your model, set the convert_tmx_to_sql variable to False so the content is not re-imported.

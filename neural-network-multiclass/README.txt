Language identifier - Neural network multiclass model

The purpose of this model is to predict the language in which a sentence/document given by the user is written:

-English
-French
-Italian
-German
-Spanish

This is also called as classification with multiple classes, as opposed to the binary models where only 2 classes can be found.

Python files:
utils.py >> helper file to get the methods called from execute.py
execute.py >> main file that will train the model, get the accuracy with the testing set and predict your given sentence (folder path at line 195 to read text files and predict the language, as in the examples given under the "predictions" folder)

NOTE: if you haven't built the SQL database yet (it's done automatically by running the execute.py file), you'll need to place a TMX file, or a set of TMX files, into the folder "dataset" and set the variable convert_tmx_to_sql to True (line 11 of execute.py). If you have already built the SQL database, you can skip this task (in order not to reimport into the database the same stuff) by setting the variable convert_tmx_to_sql to False (again, line 11 of execute.py). But take into account that lines 85-98 from execute.py depend on the size of your data. More information below.

The "dataset" directory should include either a TMX file (or a set of TMX files) or a database. For this task, I used a set of 1,357 TMX files (more than 4 million translation units in many language pairs) and it resulted in a 1GB database, but you can use the resources you would like. I am leaving this folder empty because GitHub has a limit for file uploads, but the TMX files used for this test (around 90% accuracy) can be found here: http://wt-public.emm4u.eu/Resources/DGT-TM-2019/Vol_2018_1.zip. After downloading the ZIP, unzip it and place it under the "dataset" directory (the TMX files can be under the folder "Vol_2018_1" since the script will read the subfolders too), make sure the convert_tmx_to_sql variable is set to True (line 11 of execute.py) and run the file. For the next usages, if you are not importing anything else and you just want to try out your model, set the convert_tmx_to_sql variable to False (again, line 11 of execute.py) so the content is not re-imported.

NOTE: the TMX files are currently modified before being parsed for a better performance, so keep a copy of these before processing them.
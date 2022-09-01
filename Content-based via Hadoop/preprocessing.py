import pandas as pd
import numpy as np
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #tokenizes a collection of words extracted from a text doc
from ast import literal_eval #This evaluates whether an expresion is a Python datatype or not

data = pd.read_csv('/content/drive/MyDrive/coursea_data-1.csv')
print(data.shape)
data.head()

#There are many null values
data.isnull().sum()

#Lets convert all Null values into 'missing value'
data = data.fillna('missing value')

"""# Data Preprocessing

**Things to do:**
* Impute all missing values
* Extract only relevant columns
* Convert all columns into lower case
* Split all names into comma separated
* Combine director, writer, actor names, production company into 1 word respectively this will be used for text extraction
"""

data.columns

#Extract relevant columns that would influence a movie's rating based on the content.

#Due to memory issue using just 3k data. You can try this code on Google Colabs for better performance
# data1 = data[['course_title','course_organization','course_Certificate_type','course_rating',
#               'course_difficulty','course_students_enrolled']].head(3000)

data1 = data[['course_title','course_rating',
              'course_difficulty','course_students_enrolled']]            
data1.head()

"""Remember the more columns you extract here more are the chances of overfitting as movies recommended will also take into account director, writer, production_company and et all. These features may be irrelevant to a user who wants to be recommended a movie based on his preferences."""

data1.isnull().sum()

#Impute all missing values
data1 = data1.fillna('missing value')

#Convert all columns into lower case
data1 = data1.applymap(lambda x: x.lower() if type(x) == str else x)
data1.head()

data2 = data1[['course_title', 'course_difficulty']]

data2.head()

# empty list
my_list = []
data2['course_title_keywords'] = ''

import nltk
nltk.download('stopwords')
nltk.download('punkt')

for index, row in data1.iterrows():
    CourseTitle = row['course_title']
    
    #instantiating Rake by default it uses English stopwords from NLTK and discards all punctuation chars

    r = Rake()
    
    #extract words by passing the text

    r.extract_keywords_from_text(CourseTitle)
    
    #get the dictionary with key words and their scores
    keyword_dict_scores = r.get_word_degrees()
    
    #assign keywords to new columns
    # row['course_title_keywords'] = list(keyword_dict_scores.keys())
    data2.at[index,'course_title_keywords']  =  list(keyword_dict_scores.keys())

data2.drop("course_title", axis=1, inplace=True)

data2.head()

#Bag of words for courses
data2['bow'] = ''
columns = data2.columns
for index, row in data2.iterrows():
    words = ''
    for col in columns:
        words = words + ' '.join(row[col])+ ' '
        row['bow'] = words

data2.head()

jobs_DF = pd.read_csv('/content/drive/MyDrive/data job posts.csv')
jobs_DF.head()

data_3 = jobs_DF[['JobDescription', 'JobRequirment']]

data_3 = data_3.fillna('missing value')
data_3 = data_3.applymap(lambda x: x.lower() if type(x) == str else x)
data_3.head()

#Create a empty list Keywords
data_3['keywords'] = ''

#Loop across all rows to extract all keywords from description
for index, row in data_3.iterrows():
    JobDescription = row['JobDescription']
    JobRequirment =  row['JobRequirment']
    
    #instantiating Rake by default it uses English stopwords from NLTK and discards all punctuation chars
    r1 = Rake()
    r2 = Rake()
    
    #extract words by passing the text

    r1.extract_keywords_from_text(JobDescription)
    r2.extract_keywords_from_text(JobRequirment)
    
    #get the dictionary with key words and their scores

    keyword_dict_scores1 = r1.get_word_degrees() 

    keyword_dict_scores2 = r2.get_word_degrees()
    
    # #assign keywords to new columns
    # row['keywords'] = list(keyword_dict_scores2.keys())+ list(keyword_dict_scores1.keys())
    data_3.at[index,'keywords']  = list(keyword_dict_scores2.keys())+ list(keyword_dict_scores1.keys())
data_3.drop("JobRequirment",  axis=1, inplace=True)
data_3.drop("JobDescription", axis=1, inplace=True)

data_3['bow'] = ''
columns = data_3.columns

for index, row in data_3.iterrows():
    words = ''
    for col in columns:
        words = words + ' '.join(row[col])+ ' '
        row['bow'] = words




#data_3.head()

Jobs = data_3[['bow']]
Courses = data2[['bow']]

jobs_DF.set_index('Title', inplace = True)

indices = pd.Series(jobs_DF.index)

#while True:
jobTitle = input("Enter the Job title as written in excel file:")

with pd.ExcelWriter('mapInput.xlsx') as writer:
     Courses.to_excel(writer, sheet_name = 'Sheet_1')

import os

try:
     os.system('cmd /k "start-dfs.sh"')
     os.system('cmd /k "jps"')
     os.system('cmd /k "start-yarn.sh"')
     os.system('cmd /k "hdfs dfs -mkdir /input"')
     os.system('cmd /k "hdfs dfs -copyFromLocal /mapInput.csv /input"')
     os.system('cmd /k "hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.10.2.jar"')
     os.system('cmd /k "-mapper mapper.py -reducer -reducer.py -input /input/mapInput.csv -output /output1"')          

except:     







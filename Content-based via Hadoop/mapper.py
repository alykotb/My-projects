import re
import sys
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import jobTitle
from preprocessing import indices
from preprocessing import Jobs


csvData = csv.reader(sys.stdin)
Courses = pd.read(csvData)

index = indices[indices == jobTitle].index[0]
Job = Jobs.iloc[index]
CoursesPlusJob = Courses.append(Job, ignore_index = True)
count = CountVectorizer()
count_matrix = count.fit_transform(CoursesPlusJob['bow'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
# cosine_sim
score_series = pd.Series(cosine_sim[len(cosine_sim)]).sort_values(ascending = False)
splitScores = score_series.iloc[1:10+1].index

# write in a csv file
with pd.ExcelWriter('mapOutput.xlsx') as writer:
     splitScores.to_excel(writer, sheet_name = 'Sheet_1')
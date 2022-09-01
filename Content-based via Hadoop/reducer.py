import re
import sys
import pandas as pd

from preprocessing import data1

topCoursesMap = pd.DataFrame()
csvData = csv.reader(sys.stdin)
topCoursesMap = pd.read(csvData)
#for frame in sys.stdin:
 #   pd.concat([topCoursesMap,frame], axis=0)

scoreSerires = topCoursesMap.sort_values(ascending = False)
top_10_indexes = list(scoreSerires .iloc[1:10+1].index)

 for i in top_10_indexes:
     print(data1.iloc[i]['course_title'])
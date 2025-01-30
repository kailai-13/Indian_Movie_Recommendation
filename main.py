import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

mv_d=pd.read_csv('indian movies.csv')
mv_d=mv_d[25000:]
s_fea=mv_d[['Genre','Language','Rating(10)','Timing(min)','Year']]
for f in s_fea:
  mv_d[f]=mv_d[f].fillna('')

com = mv_d['Genre'] + ' ' + mv_d['Language'] + ' ' + mv_d['Rating(10)'] + ' ' + mv_d['Timing(min)'] + ' ' + mv_d['Year']

vectorizer=TfidfVectorizer()
f_vectors=vectorizer.fit_transform(com)

sim=cosine_similarity(f_vectors)

mv_m=input('Enter Your Fav Movie Name:')

list_all=mv_d['MovieName'].tolist()

find=difflib.get_close_matches(mv_m,list_all)

cmatch=find[0]

index_m=mv_d[mv_d.MovieName==cmatch]['index'].values[0]

sim_score=list(enumerate(sim[index_m]))

sort_sim_m=sorted(sim_score,key=lambda x:x[1] ,reverse=True)
print('Movies suggested for you : \n')

i = 1

for movie in sort_sim_m:
  index = movie[0]
  title_from_index = mv_d[mv_d.index==index]['Movie Name'].values[0]
  rating=mv_d[mv_d.index==index]['Rating(10)'].values[0]
  time=mv_d[mv_d.index==index]['Timing(min)'].values[0]
  if (i<30):
    print(i, '.',title_from_index,'-',rating,'Duration:',time)
    i+=1
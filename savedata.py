#Download data
import pandas
import MySQLdb
import nltk

def getdata():
    sql='select * from post_clean_content limit 100000,2000'
    db = MySQLdb.connect(host="localhost", user="query", passwd="1234", db="blogs")
    df = pandas.read_sql_query(sql, db, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None)
    print("read data")
    for index, row in df.iterrows():
        if(index%100000 == 0):print(index)
        words=[x.lower() for x in nltk.word_tokenize(row['content'])]
        if(not set(['i', 'you', 'me', 'my', 'your', 'mine', 'our', 'ours', 'yours']).intersection(set(words))):
            df.drop(index, inplace=True)
    return df

data = getdata()
print("processing done")
data.to_json(r'pandasdev.txt')
print("saved")
#getdata().to_csv(r'pandas.txt')

#read json
#o2 = pandas.read_json('pandas2.txt')
#for index, row in o.iterrows():
#    print(row['content'])
    
#o = pandas.read_csv('pandas.txt')
#for index, row in o.iterrows():
#    print(row['content'])

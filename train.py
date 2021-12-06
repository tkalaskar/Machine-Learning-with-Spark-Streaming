import time
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession, functions as func
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
import pickle

sc = SparkContext("local[2]", "sentiment")
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sq=SQLContext(sc)





def God_bless(rdd):
	if(rdd.isEmpty()!=True):
		global classifier
		rdd1=rdd.flatMap(lambda x: x.split('}'))
		rdd1=rdd1.filter(lambda x: x!='')
		#print(rdd1.collect())
		rdd2=rdd1.map(lambda x: (int(x.split('"feature0": ')[1][0]),(x.split('"feature1": ')[1][1:-1]).lower().strip()))
		df=sq.createDataFrame(rdd2,schema=['Sentiment','Text'])
		df=df.withColumn('Text',func.regexp_replace('Text',r'http\S+',''))
		df=df.withColumn('Text',func.regexp_replace('Text','@\w+',''))
		df=df.withColumn('Text',func.regexp_replace('Text','#',''))
		df=df.withColumn('Text',func.regexp_replace('Text',':',' '))
		df=df.withColumn('Text',func.regexp_replace('Text',r'[^\w ]',' '))
		df=df.withColumn('Text',func.regexp_replace('Text',r'[\d]',''))
		df=df.withColumn('Text',func.regexp_replace('Text',r'\b[a-zA-Z]\b',''))
		df=df.withColumn('Text',func.regexp_replace('Text',r'\b[a-zA-Z][a-zA-Z]\b',''))
		df=df.withColumn('Text',func.regexp_replace('Text',' +',' '))
		df=df.withColumn('Text',func.regexp_replace('Text','^\s+|\s+$',''))
		
		tokenizer=Tokenizer(inputCol="Text",outputCol="Senti_Words")
		tokenized_df=tokenizer.transform(df)
		
		remover=StopWordsRemover(inputCol="Senti_Words", outputCol="Meaningful_Words")
		filtered_df=remover.transform(tokenized_df)
		
		hashTF=HashingTF(inputCol="Meaningful_Words",outputCol="Features")
		numeric_df=hashTF.transform(filtered_df).select('Sentiment','Meaningful_Words','Features')
		
		model_training1(numeric_df)

def model_training1(tf_df):
	rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="Sentiment", seed=42,leafCol="Features")
	model = rf.fit(tf_df)
	pickle.dump(model, open('model.pkl', 'wb'))

		


lines = ssc.socketTextStream('localhost',6100)
lines.foreachRDD(lambda rdd :God_bless(rdd))






	



ssc.start()
ssc.awaitTermination()
ssc.stop()

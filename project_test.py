import time
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession, functions as func
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

sc = SparkContext("local[2]", "sentiment")
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sq=SQLContext(sc)

def God_bless(rdd):
	if(rdd.isEmpty()!=True):
		
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
		
		X=np.array(numeric_df.select("Features").collect())
		Y=np.array(numeric_df.select("Sentiment").collect())
		
		model_prediction1(X,Y)

def model_prediction1(X_test,Y_test):
	serialized_model = open("/home/pes1ug19cs543/BD/model.pickle", "rb")
	model = pickle.load(serialized_model)
	serialized_model.close()
    	
	X_test=X_test.reshape(X_test.shape[0], (X_test.shape[1]*X_test.shape[2]))
	Y_preds = list(model.predict(X_test))
	print("Test Accuracy 1      : {}".format(accuracy_score(Y_test.reshape(-1), Y_preds)))



lines = ssc.socketTextStream('localhost',6100)
lines.foreachRDD(lambda rdd :God_bless(rdd))





	



ssc.start()
ssc.awaitTermination()
		


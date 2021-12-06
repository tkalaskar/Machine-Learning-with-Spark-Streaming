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
from sklearn.cluster import MiniBatchKMeans
import pickle

sc = SparkContext("local[2]", "sentiment")
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sq=SQLContext(sc)
global classifier
classifier = SGDClassifier(loss='log',random_state=0)
global cluster
cluster=MiniBatchKMeans(n_clusters=5, random_state=123)


def rfclassifier(rdd):
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
		
		remover=StopWordsRemover(inputCol="Senti_Words", outputCol="MeaningfulWords")
		filtered_df=remover.transform(tokenized_df)
		
		hashTF=HashingTF(inputCol="MeaningfulWords",outputCol="Features")
		num_df=hashTF.transform(filtered_df).select('Sentiment','MeaningfulWords','Features')
		
		X=np.array(num_df.select("Features").collect())
		Y=np.array(num_df.select("Sentiment").collect())
		
		model_training1(X,Y)
		#model_training2(X,Y)
		model_training2(num_df)

def model_training1(X_train,Y_train):
	global classifier
	
	X_train=X_train.reshape(X_train.shape[0], (X_train.shape[1]*X_train.shape[2]))
	classifier.partial_fit(X_train,Y_train.ravel(),classes=[0,4])
	pickle.dump(classifier, open( "/home/pes1ug19cs054/Desktop/Project/model.pickle", "wb" ) )

def model_training2(tf_df):
	rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="Sentiment", seed=42,leafCol="Features")
	model = rf.fit(tf_df)
	pickle.dump(model, open('model.pkl', 'wb'))
	pickle.dump(classifier, open( "/home/pes1ug19cs054/Desktop/Project/model.pickle", "wb" ) )
	
		






lines = ssc.socketTextStream('localhost',6100)
lines.foreachRDD(lambda rdd :rfclassfier(rdd))





	



ssc.start()
ssc.awaitTermination()

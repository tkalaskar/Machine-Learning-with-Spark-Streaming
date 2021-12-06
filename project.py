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
global classifier
classifier = SGDClassifier(loss='log',random_state=0)
global model



def preprocessing(rdd):
	if(rdd.isEmpty()!=True):
		global classifier
		rdd1=rdd.flatMap(lambda x: x.split('}'))
		rdd1=rdd1.filter(lambda x: x!='')
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
		train(numeric_df)		
def train(tf_df):
	#global model
	#X=np.array(tf_df.select("Features").collect())
	#Y=np.array(tf_df.select("Sentiment").collect())
	#X_train, Y_train = train_test_split(X,Y, train_size=1, random_state=0)
	#X=X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))
	#X_test=X_test.reshape(X_test.shape[0], (X_test.shape[1]*X_test.shape[2]))
	#print(X_train)
	#print(Y_train)
	#classifier.partial_fit(X,Y.ravel(),classes=[0,4])
	#Y_preds = list(classifier.predict(X_test))
	#print("Test Accuracy: {}".format(accuracy_score(Y_test.reshape(-1), Y_preds)))
	print(1)
	dividedData = tf_df.randomSplit([0.7, 0.3])
	trainingData = dividedData[0] #index 0 = data training
	testingData = dividedData[1] #index 1 = data testing
	lr = LogisticRegression(labelCol="Sentiment", featuresCol="Features", 
                        maxIter=10, regParam=0.01)
	model = lr.fit(trainingData)
	





lines = ssc.socketTextStream('localhost',6100)
lines.foreachRDD(lambda rdd :preprocessing(rdd))





	



ssc.start()
ssc.awaitTermination()
scc.stop()

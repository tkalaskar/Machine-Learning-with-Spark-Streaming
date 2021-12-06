import time
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession, functions as func
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.ml.feature import HashingTF,IDF
from pyspark.ml.feature import NGram
from pyspark.ml.feature import HashingTF,IDF


sc = SparkContext("local[2]", "sentiment")
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sq=SQLContext(sc)

def preprocessing(rdd):
	if(rdd.isEmpty()!=True):
		
		rdd1=rdd.flatMap(lambda x: x.split('}'))
		rdd1=rdd1.filter(lambda x: x!='')
		rdd2=rdd1.map(lambda x: (int(x.split('"feature0": ')[1][0]),(x.split('"feature1": ')[1][1:-1]).lower()))
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
		#df= df.withColumn('Text', 'Text'.rstrip())
		tokenizer=Tokenizer(inputCol="Text",outputCol="SentWords")
		tokenized_df=tokenizer.transform(df)
		remover=StopWordsRemover(inputCol="SentWords", outputCol="filtered")
		filtered_df=remover.transform(tokenized_df)
		ngram=NGram(n=2, inputCol="filtered", outputCol="2gram")
		gram_df=ngram.transform(filtered_df)
		hashingtf=HashingTF(inputCol="2gram", outputCol="tf", numFeatures=20000)
		tf_df=hashingtf.transform(gram_df)
		print(tf_df.show())


		


	
	


lines = ssc.socketTextStream('localhost',6100)
lines.foreachRDD(lambda rdd :preprocessing(rdd))
ssc.start()
ssc.awaitTermination()

import numpy as np
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
from pyspark.sql import Row,SQLContext,SparkSession,functions as F
from pyspark.sql import Row
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

if __name__ == "__main__":
	sc= SparkContext(master="local[2]",appName="trial")
	ssc = StreamingContext(sc,10)
	spark = SparkSession(sc)
	lines= ssc.socketTextStream("localhost", 6100)
	sql=SQLContext(sc)
	

	word=lines.flatMap(lambda line: line.split("\n")

	def readdata(rd):
		f0=[]
		f1=[]
		#print(rd)
		df= spark.read.json(rd)
		#df=sqlContext.createDataFrame(rd)
		f=df.collect()
		for i in f:
			for k in i:
				f0.append(k[0])
				f1.append(k[1].strip())
		if(len(f0)!=0 and len(f1)!=0):
			x=sql.createDataFrame(zip(f0,f1),schema=['Sentiment','Tweet'])
			#x=data.to_csv(r'\home/pes1ug19cs054/Desktop/Spam/textfile.txt', header=None, index=None, sep=' ', mode='w')
			y=x.select('Tweet').collect()
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'http\S+',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','@',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','#',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',':',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'[^\w ]',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'[\d]',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'\b[a-zA-Z]\b',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',r'\b[a-zA-Z][a-zA-Z]\b',''))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet',' +',' '))
			x=x.withColumn('Tweet',F.regexp_replace('Tweet','^\s+|\s+$',''))
			data=x.to_csv(r'\home/pes1ug19cs054/Desktop/Spam/textfile.txt', header=None, index=None, sep=' ', mode='w')
			

			stop_words = set(stopwords.words('english'))
			word_tokens = word_tokenize(data)
			filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
			filtered_sentence = []
 
			for w in word_tokens:
    			if w not in stop_words:
        			filtered_sentence.append(w)



	rdd=word.foreachRDD(readdata)
	#rdd=word.map(lambda x: json.loads(x))	
	#r=json.loads(lines)
	print("new batch")
	ssc.start()
	ssc.awaitTermination()
	ssc.stop()


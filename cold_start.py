# Import the required libraries
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import col, expr

spark = SparkSession.builder.appName("My_Session").getOrCreate()

data = spark.read.parquet('interactions.parquet')
print('Original Dataset: ')
data.show()
genre = spark.read.json('goodreads_book_genres_initial.json')

# Convert genre dataframe to parquet
# genre.createOrReplaceTempView('genre')
# genre.write.parquet('genres.parquet')

train,val,test = data.randomSplit(weights=[0.6, 0.2, 0.2], seed=35)

# Hyperparameter Tuning
for l in np.arange(0.05,0.15,0.01):
  for r in np.arange(10,50,5):

    # Create ALS Model
    als = ALS(rank=r,regParam=l, userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop', nonnegative = True)

    # Train the model
    model = als.fit(train)

    evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

    predictions = model.transform(val)
    rmse = evaluator.evaluate(predictions)
    #predictions.show()

    predictions1 = model.transform(test)
    rmse1 = evaluator.evaluate(predictions1)
    print('Rank: {} \tLambda: {:.6f} \tRMSE Validation: {:.6f} \tTest Loss: {:.6f}'.format(
          r,
          l,
          rmse,
          rmse1
          ))
  #print("RMSE for Validation Set = ", rmse)
  #print("RMSE for Test Set = ", rmse1)
  # predictions = np.round(predictions)
  #predictions1.show()


r = 10
l = 0.14
als = ALS(rank = r, regParam = l, userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', nonnegative = True)

# Train the model
model = als.fit(train)

# genre.createOrReplaceTempView('genre')
genre = spark.sql('SELECT * FROM data INNER JOIN genre ON data.book_id = genre.book_id')
print('Genre data is:')
genre.show()
genre.printSchema()

print('Training Set:')
train.show()
print('Validation Set:')
val.show()
print('Test Set:')
test.show()

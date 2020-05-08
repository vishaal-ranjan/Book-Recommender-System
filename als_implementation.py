import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("My_Session").getOrCreate()

# Load the training, validation and test sets
train = spark.read.parquet('training_set.parquet')
train = train.select(train['user_id'], train['book_id'], train['rating'])

val = spark.read.parquet('validation_set.parquet')
val = val.select(val['user_id'], val['book_id'], val['rating'])

test = spark.read.parquet('test_set.parquet')
test = test.select(test['user_id'], test['book_id'], test['rating'])

for l in np.arange(0.05,0.15,0.01):
  for r in np.arange(10,100,10):

    # Create ALS Model
    als = ALS(rank=r,regParam=l,userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop', nonnegative = True)

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

user_recs = model.recommendForAllUsers(20).show(10)

# print('Training Set: ')
# train.show()
# print(train.count())

# print('Validation Set: ')
# val.show()
# print(val.count())

# print('Test Set: ')
# test.show()
# print(test.count())

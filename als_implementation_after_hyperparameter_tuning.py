# Import the required libraries
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import pyspark.sql.functions as func
from pyspark.sql.functions import round, col, expr
from pyspark import SparkContext

spark = SparkSession.builder.appName("My_Session").getOrCreate()

# SparkContext.setSystemProperty('spark.executor.memory', '8g')
# sc = SparkContext("local", "App Name")
# sc._conf.getAll()

# Load the training, validation and test sets
train = spark.read.parquet('training_set.parquet')
train = train.select(train['user_id'], train['book_id'], train['rating'])

val = spark.read.parquet('validation_set.parquet')
val = val.select(val['user_id'], val['book_id'], val['rating'])

test = spark.read.parquet('test_set.parquet')
test = test.select(test['user_id'], test['book_id'], test['rating'])

# See some statistics about the train, validation and test data
print('Statistics for Training Data: ')
train.describe().show()
print('Statistics for Validation Data: ')
val.describe().show()
print('Statistics for Test Data: ')
test.describe().show()

# After doing the hyperparameter tuning, ideal values for rank and regParam are: rank = 50 and regParam = 0.09
r = 50
l = 0.09
als = ALS(rank = r, regParam = l, userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop', nonnegative = True)

# Train the model
model = als.fit(train)

# RMSE value evalutation (Regression Metric)
evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

# Prediction of rating for validation set
predictions = model.transform(val)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"]))
rmse = evaluator.evaluate(predictions)
#predictions.show()

# Prediction of rating for test set
predictions1 = model.transform(test)
predictions1 = predictions1.withColumn("prediction", func.round(predictions1["prediction"]))
rmse1 = evaluator.evaluate(predictions1)
print('Rank: {} \tLambda: {:.6f} \tRMSE Validation: {:.6f} \tTest Loss: {:.6f}'.format(
        r,
        l,
        rmse,
        rmse1
        ))

print('Predictions of Ratings on the Validation Data: ')
predictions.show()
print('Schema for validation set predictions: ')
predictions.printSchema() 

print('Predictions of Ratings on the Test Data: ')
predictions1.show()
print('Schema for test set predictions: ')
predictions1.printSchema()

# Recommend books for all users
user_recs = model.recommendForAllUsers(500)

# user_recs = user_recs.withColumn("recommendations.rating", func.round(user_recs["recommendations.rating"]))

print('Top 500 Recommendations for each user: ')
user_recs1 = model.recommendForAllUsers(500).selectExpr('user_id', 'explode(recommendations)').show()
user_recs1 = model.recommendForAllUsers(500).selectExpr('user_id', 'explode(recommendations)')

# Ordering the recommendations by user_id
user_recs1.createOrReplaceTempView('user_recs1')
display = spark.sql('SELECT * FROM user_recs1 ORDER BY user_id')
print('After being ordered by user_id: ')
display.show(50)

# The recommendations for a particular user
user = 22000
print('Recommendations for user_id', user, 'is: ')
display = spark.sql('''SELECT * FROM user_recs1 WHERE user_id =  22000''')
display.show()
print(type(user_recs))
print(user_recs.printSchema())

# actual_val = val.groupBy("user_id").agg(expr("collect_set(book_id) as books"))
# pred_val = user_recs.select('user_id','recommendations.book_id')
# output_val = pred_val.join(actual_val,['user_id']).select('book_id','books')
# metrics_val = RankingMetrics(output_val.rdd)
# result_val = metrics_val.meanAveragePrecision
# print('The MAP is:', result_val)

# Mean Average Precision (Ranking Metric)
actual_test = test.groupBy("user_id").agg(expr("collect_set(book_id) as books"))
pred_val = user_recs.select('user_id','recommendations.book_id')
output_test = pred_val.join(actual_test,['user_id']).select('book_id','books')
metrics_test = RankingMetrics(output_test.rdd)
result_test = metrics_test.meanAveragePrecision
print('The MAP is:', result_test)

# print('Training Set: ')
# train.show()
# print(train.count())

# print('Validation Set: ')
# val.show()
# print(val.count())

# print('Test Set: ')
# test.show()
# print(test.count())
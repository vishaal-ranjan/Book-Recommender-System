# Import the required libraries
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql.functions import col, expr

spark = SparkSession.builder.appName("My_Session").getOrCreate()

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

# Hyperparameter Tuning
# for l in np.arange(0.05,0.15,0.01):
#   for r in np.arange(10,100,10):

#     # Create ALS Model
#     als = ALS(rank=r,regParam=l,userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop', nonnegative = True)

#     # Train the model
#     model = als.fit(train)

#     evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

#     predictions = model.transform(val)
#     rmse = evaluator.evaluate(predictions)
#     #predictions.show()

#     predictions1 = model.transform(test)
#     rmse1 = evaluator.evaluate(predictions1)
#     print('Rank: {} \tLambda: {:.6f} \tRMSE Validation: {:.6f} \tTest Loss: {:.6f}'.format(
#           r,
#           l,
#           rmse,
#           rmse1
#           ))
#   #print("RMSE for Validation Set = ", rmse)
#   #print("RMSE for Test Set = ", rmse1)
#   # predictions = np.round(predictions)
#   #predictions1.show()

# After doing the hyperparameter tuning, ideal values for rank and regParam are: rank = 50 and regParam = 0.09
r = 50
l = 0.09
als = ALS(rank = r, regParam = l, userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop', nonnegative = True)

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

# np.round(predictions1)

print('Predictions of Ratings on the Test Data: ')
predictions1.show()
print('Schema for test set predictions: ')
predictions1.printSchema()

# Recommend books for all users
user_recs = model.recommendForAllUsers(5)

# print('Top 5 Recommendations for each user: ')
# # user_recs = model.recommendForAllUsers(5).selectExpr('user_id', 'explode(recommendations)').show()
# user_recs = model.recommendForAllUsers(5).selectExpr('user_id', 'explode(recommendations)')
# user_recs.show()
# # user_recs = model.recommendForAllUsers(5)
# user_recs.createOrReplaceTempView('user_recs')
# display = spark.sql('SELECT * FROM user_recs ORDER BY user_id')
# print('After being ordered by user_id: ')
# display.show(50)

# # The recommendations for a particular user
# user = 22000
# print('Recommendations for user_id', user, 'is: ')
# display = spark.sql('''SELECT * FROM user_recs WHERE user_id =  22000''')
# display.show()
# print(type(user_recs))
# print(user_recs.printSchema())
# # print(display.first())
# # print("RMSE = %s" % user_recs.rootMeanSquaredError)


actual_val = val.groupBy("user_id").agg(expr("collect_set(book_id) as books"))
pred_val = user_recs.select('user_id','recommendations.book_id')
output_val = pred_val.join(actual_val,['user_id']).select('book_id','books')
metrics_val = RankingMetrics(output_val.rdd)
result_val = metrics_val.meanAveragePrecision
print('The MAP is:', result_val)

# result_val = metrics_val.meanAveragePrecision
# print('The Mean Precision at k is:', result_val)

# result_val = metrics_val.nDCGAt(5)
# print('The nDCG is:', result_val)

# print('Training Set: ')
# train.show()
# print(train.count())

# print('Validation Set: ')
# val.show()
# print(val.count())

# print('Test Set: ')
# test.show()
# print(test.count())
#Importing the required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
spark = SparkSession.builder.appName("My_Session").getOrCreate()

# Load the goodreads_interactions dataset as a dataframe
data = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', schema='user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
print('Original dataset count: ', data.count())
data.createOrReplaceTempView('data')

# Removing the entries where rating = 0
results = spark.sql('SELECT * FROM data WHERE rating != 0')
results.createOrReplaceTempView('results')

# Removing users which have fewer than 10 interactions
result1 = spark.sql('SELECT * FROM results WHERE user_id IN (SELECT user_id FROM results GROUP BY user_id HAVING COUNT(user_id) > 10)')
print('After filtering interactions < 10: ', result1.count())
result1.show()
# result1 = spark.sql('SELECT * FROM results GROUP BY user_id ORDER BY COUNT(is_read)')

# Selecting 1% of all users
result1.createOrReplaceTempView('result1')
result1 = spark.sql('SELECT * FROM result1 WHERE user_id%100 = 0')
print('After filtering user_id: ', result1.count())

# # Convert goodreads_interactions dataframe to parquet file
# result1.write.parquet('interactions.parquet')
result2 = spark.read.parquet('interactions.parquet')

# We will randomly split the interactions dataset to train, validation and test sets in a 60:20:20 ratio
train,val,test = result2.randomSplit(weights=[0.6, 0.2, 0.2], seed=45)

print('Original Training Count: ', train.count())

# Sort the validation set in ascending order of user_id field
result3 = val.orderBy(val.user_id.asc())
# result3.show()

# Add a Sequential ID column 'row_num' to the validation set
result3= result3.withColumn("new_column",lit("ABC"))
w = Window().partitionBy('new_column').orderBy(lit('A'))
df = result3.withColumn("row_num", row_number().over(w)).drop("new_column")

# We select all the odd numbered rows from the validation set
df.createOrReplaceTempView('df')
df = spark.sql('SELECT * FROM df WHERE row_num%2 = 1')


# We will drop the row_num column from the validation set
df = df.drop('row_num')

# We append the odd_numbered rows of validation set to the training set
train = train.union(df)
print('New Training Count: ', train.count())

# The new validation set will contain all the even numbered rows from the original test set
val = spark.sql('SELECT * FROM df WHERE row_num%2 = 0')

# We drop the row_num column from the validation set
val = val.drop('row_num')
print('Validation Count: ', val.count())


# Sort the test set in ascending order of user_id field
result4 = test.orderBy(test.user_id.asc())

# Add a Sequential ID column 'row_num' to the test set
result4= result4.withColumn("new_column",lit("ABC"))
w1 = Window().partitionBy('new_column').orderBy(lit('A'))
df1 = result4.withColumn("row_num", row_number().over(w1)).drop("new_column")

# We select all the odd numbered rows from the test set
df1.createOrReplaceTempView('df1')
df1 = spark.sql('SELECT * FROM df1 WHERE row_num%2 = 1')

# We will drop the row_num column from the test set
df1 = df1.drop('row_num')

# We append the odd_numbered rows of test set to the training set
train = train.union(df1)
print('New Training Count: ', train.count())

# The new test set will contain all the even numbered rows from the original test set
test = spark.sql('SELECT * FROM df1 WHERE row_num%2 = 0')

# We drop the row_num column from the test set
test = test.drop('row_num')
print('Test Count: ', test.count())

# Convert train, val, test dataframes to parquet files
train.write.parquet('training_set.parquet')
val.write.parquet('validation_set.parquet')
test.write.parquet('test_set.parquet')

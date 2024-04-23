import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

def clean_data(df):
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName('cs643_wine_prediction') \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')
if len(sys.argv) > 3:
    sys.exit(-1)
elif len(sys.argv) > 1:
    input_path = sys.argv[1]
    
    if not("/" in input_path):
        input_path = "data/csv/" + input_path
    model_path="/code/data/model/testdata.model"
    print("Test data for Input file ")
    print(input_path)
else:
    current_dir = os.getcwd() 
    print(current_dir)
    input_path = os.path.join(current_dir,"cs643-pa2\data\csv\testdata.csv")
    model_path= os.path.join(current_dir, "cs643-pa2\data\model\testdata.model")
df = (spark.read
        .format("csv")
        .option('header', 'true')
        .option("sep", ";")
        .option("inferschema",'true')
        .load(input_path))

df1 = clean_data(df)
all_features = ['fixed acidity',
                    'volatile acidity',
                    'citric acid',
                    'chlorides',
                    'total sulfur dioxide',
                    'density',
                    'sulphates',
                    'alcohol',
                ]
rf = PipelineModel.load(model_path)
predictions = rf.transform(df1)
print(predictions.show(5))
results = predictions.select(['prediction', 'label'])
evaluator = MulticlassClassificationEvaluator(
                                        labelCol='label', 
                                        predictionCol='prediction', 
                                        metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Wine prediction model Test Accuracy = ', accuracy)
metrics = MulticlassMetrics(results.rdd.map(tuple))
print('Wine prediction model for Weighted f1 score = ', metrics.weightedFMeasure())
sys.exit(0)

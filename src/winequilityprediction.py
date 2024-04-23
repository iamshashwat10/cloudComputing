
import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_data(df):
    # cleaning header 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .appName('cs643_wine_prediction') \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) == 3:
        input_path = sys.argv[1]
        valid_path = sys.argv[2]
        output_path = sys.argv[3] + "testmodel.model"
    else:
        input_path = "s3://bucketasgnt2/TrainingDataset.csv"
        valid_path = "s3://bucketasgnt2/ValidationDataset.csv"
        output_path= "s3://bucketasgnt2/testmodel.model"

    
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    train_data_set = clean_data(df)

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(valid_path))
    
    valid_data_set = clean_data(df)

    all_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'residual sugar',
                        'chlorides',
                        'free sulfur dioxide',
                        'total sulfur dioxide',
                        'density',
                        'pH',
                        'sulphates',
                        'alcohol',
                        'quality',
                    ]
    assembler = VectorAssembler(inputCols=all_features, outputCol='features')
    
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    train_data_set.cache()
    valid_data_set.cache()
    
    rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=150, maxBins=8, maxDepth=15, seed=150,impurity='gini')
    
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_data_set)

    predictions = model.transform(valid_data_set)

 
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                        predictionCol='prediction', 
                                        metricName='accuracy')

    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of wine prediction model= ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())
 
    cvmodel = None
    paramGrid = ParamGridBuilder() \
            .addGrid(rf.maxBins, [9, 8, 4])\
            .addGrid(rf.maxDepth, [25, 6 , 9])\
            .addGrid(rf.numTrees, [500, 50, 150])\
            .addGrid(rf.minInstancesPerNode, [6])\
            .addGrid(rf.seed, [100, 200, 5043, 1000])\
            .addGrid(rf.impurity, ["entropy","gini"])\
            .build()
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

  
    cvmodel = crossval.fit(train_data_set)
    
    model = cvmodel.bestModel
    print(model)
    
    predictions = model.transform(valid_data_set)
    results = predictions.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy1 of wine prediction model = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())

    model_path =output_path
    model.write().overwrite().save(model_path)
    sys.exit(0)
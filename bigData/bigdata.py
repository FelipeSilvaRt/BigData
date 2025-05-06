from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Inicia sessão Spark
spark = SparkSession.builder.appName("BigData_ML_Example").getOrCreate()

# 2. Carrega os dados
data = spark.read.csv("seus_dados.csv", header=True, inferSchema=True)

# 3. Pré-processamento
# Supondo que 'label' é a coluna alvo e as demais são features
feature_cols = [col for col in data.columns if col != 'label']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Transforma a coluna label em numérica (caso necessário)
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
data = indexer.fit(data).transform(data)

# 4. Divide o dataset em treino e teste
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# 5. Modelo Árvore de Decisão
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

# 6. Modelo SVM
svm = LinearSVC(labelCol="indexedLabel", featuresCol="features", maxIter=10)
svm_model = svm.fit(train_data)
svm_predictions = svm_model.transform(test_data)

# 7. Avaliação
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

dt_accuracy = evaluator.evaluate(dt_predictions)
svm_accuracy = evaluator.evaluate(svm_predictions)

print(f"Acurácia Árvore de Decisão: {dt_accuracy}")
print(f"Acurácia SVM: {svm_accuracy}")

# 8. Encerra a sessão Spark
spark.stop()

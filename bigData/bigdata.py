from pyspark.sql import SparkSession
from pyspark.sql.functions import month, to_date
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Inicia sessão Spark
spark = SparkSession.builder.appName("Proj_Arduino").getOrCreate()

# 2. Carrega os dados
data = spark.read.csv("vendas_rio_tuor_2023.csv", header=True, inferSchema=True)

# 3. Pré-processamento

# Converte a coluna de data (caso esteja em texto)
data = data.withColumn("Data", to_date("Data", "dd/MM/yyyy"))

# Extrai o mês da data
data = data.withColumn("Mes", month("Data"))

# Transforma a coluna alvo 'Promoções' em número
label_indexer = StringIndexer(inputCol="Promoções", outputCol="indexedLabel")
data = label_indexer.fit(data).transform(data)

# Transforma a coluna 'Produto' em número (para uso como feature)
produto_indexer = StringIndexer(inputCol="Produto", outputCol="ProdutoIndex")
data = produto_indexer.fit(data).transform(data)

# Define as colunas de entrada (features)
feature_cols = [
    "Mes",
    "Quantidade Vendida",
    "Preço Unitário",
    "Total de Vendas",
    "Lucro",
    "Despesas",
    "ProdutoIndex"
]

# Monta o vetor de features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

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
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
)

dt_accuracy = evaluator.evaluate(dt_predictions)
svm_accuracy = evaluator.evaluate(svm_predictions)

print(f"Acurácia Árvore de Decisão: {dt_accuracy}")
print(f"Acurácia SVM: {svm_accuracy}")

# 8. Encerra a sessão Spark
spark.stop()

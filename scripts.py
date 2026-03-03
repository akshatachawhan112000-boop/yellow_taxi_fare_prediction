df_raw = spark.table("workspace.default.yellow_combined")
# ==========================================
# 1️⃣ LOAD PARQUET (Free Edition Safe)
# ==========================================
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Replace with your uploaded Parquet path in /FileStore/tables/
df_raw = spark.table("workspace.default.yellow_combined")

# ==========================================
# 2️⃣ DATA PREPROCESSING
# ==========================================
df = df_raw \
    .withColumn("tpep_pickup_datetime", F.to_timestamp("tpep_pickup_datetime")) \
    .withColumn("fare_amount", F.col("fare_amount").cast(DoubleType())) \
    .withColumn("trip_distance", F.col("trip_distance").cast(DoubleType())) \
    .withColumn("hour_of_day", F.hour("tpep_pickup_datetime")) \
    .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime")) \
    .withColumn("month", F.month("tpep_pickup_datetime"))

# Remove invalid or extreme values
df = df.filter(
    (F.col("fare_amount") > 0) &
    (F.col("fare_amount") < 200) &
    (F.col("trip_distance") > 0)
)

# Preview processed data
print("Preview processed data:")
df.show(5)

# ==========================================
# 3️⃣ TEMPORAL SPLIT (Jan–Jun train, Jul–Aug test)
# ==========================================
train_df = df.filter(F.col("month") <= 6)
test_df  = df.filter(F.col("month") > 6)

# Preview splits instead of counting
print("Training sample preview:")
train_df.show(5)

print("Testing sample preview:")
test_df.show(5)
# ==========================================
# 4️⃣ FEATURE ENGINEERING
# ==========================================
from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "trip_distance",
    "hour_of_day",
    "day_of_week"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# ==========================================
# 5️⃣ DEFINE ML MODELS
# ==========================================
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
    DecisionTreeRegressor,
    GeneralizedLinearRegression
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

models = {
    "LinearRegression": LinearRegression(featuresCol="features", labelCol="fare_amount"),
    "DecisionTree": DecisionTreeRegressor(featuresCol="features", labelCol="fare_amount", maxDepth=5),
    "RandomForest": RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=10, maxDepth=4),
    "GBT": GBTRegressor(featuresCol="features", labelCol="fare_amount", maxIter=10, maxDepth=4),
    "GeneralizedLinearRegression": GeneralizedLinearRegression(featuresCol="features", labelCol="fare_amount", family="gaussian")
}

evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")

# ==========================================
# 6️⃣ TRAIN & EVALUATE MODELS (Free Edition Friendly)
# ==========================================
results = {}

for name, model in models.items():
    pipeline = Pipeline(stages=[assembler, model])
    fitted_model = pipeline.fit(train_df.sample(False, 0.1, seed=42))  # Sample small subset for Free Edition
    
    predictions = fitted_model.transform(test_df.limit(5000))  # Limit test rows to avoid cluster crash
    
    rmse = evaluator.evaluate(predictions)
    results[name] = rmse
    
    print(f"{name} RMSE: {round(rmse, 2)}")

# ==========================================
# 7️⃣ BEST MODEL
# ==========================================
best_model_name = min(results, key=results.get)
print("\nBest Performing Model:", best_model_name)

# =========================================
# 9️⃣ SAVE PREDICTIONS AS GOLD TABLE
# =========================================

# If your best model variable is named differently, adjust here
cv_predictions = fitted_model.transform(test_df)

gold_table = cv_predictions.groupBy("month", "hour_of_day") \
    .agg(
        F.avg("fare_amount").alias("Actual_Avg_Fare"),
        F.avg("prediction").alias("Predicted_Avg_Fare"),
        F.count("*").alias("Trip_Count")
    )

# Save as managed table
gold_table.write.mode("overwrite").saveAsTable("yellow_gold_table")

print("\n✅ Gold table created successfully.")

# =========================================
# 🔟 LOAD GOLD TABLE
# =========================================

gold_table = spark.table("workspace.default.yellow_gold_table")

# Optional sampling (safer for Community Edition)
gold_table_sample = gold_table.sample(False, 0.1, seed=42)

# Convert to Pandas
gold_pdf = gold_table_sample.toPandas()

# Save CSV to local DBFS
csv_path = "/tmp/yellow_gold_table.csv"
gold_pdf.to_csv(csv_path, index=False)

print(f"✅ CSV saved at {csv_path}")
display(gold_pdf.head())

import java.util.Calendar
import org.apache.log4j.{Level, Logger}
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object SimpleXGBoost {
    Logger.getLogger("org").setLevel(Level.WARN)

    def main(args: Array[String]): Unit = {

        // create SparkSession
        val spark = SparkSession
        .builder()
        .appName("SimpleXGBoost Application")
        .config("spark.executor.memory", "2G")
        .config("spark.executor.cores", "4")
        .config("spark.driver.memory", "1G")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.default.parallelism", "4")
        .master("local[*]")
        .getOrCreate()
        val now=Calendar.getInstance()
        val date=java.time.LocalDate.now
        val currentHour = now.get(Calendar.HOUR_OF_DAY)
        val currentMinute = now.get(Calendar.MINUTE)
        val direct="./results/"+date+"-"+currentHour+"-"+currentMinute+"/"
        println(direct)
        ///read data from disk
        val dataset = spark.read.option("header", "true").option("inferSchema", true).csv("/home/fnargesian/opendata-organization/labeling/input/train_numeric.csv")
        val datatest = spark.read.option("header", "true").option("inferSchema", true).csv("/home/fnargesian/opendata-organization/labeling/input/test_numeric.csv")

        dataset.cache()
        datatest.cache()
        //fill NA with 0 and subsample
        val df = dataset.na.fill(0).sample(true,0.7,10)
        val df_test = datatest.na.fill(0)
        //prepare data for ML
        val header = df.columns.filter(!_.contains("Id")).filter(!_.contains("Response"))
        val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")
        val train_DF0 = assembler.transform(df)
        val test_DF0 = assembler.transform(df_test)
        println("VectorAssembler Done!")
        val train = train_DF0.withColumn("label", df("Response").cast("double")).select("label", "features")
        val test = test_DF0.withColumn("label", lit(1.0)).withColumnRenamed("Id","id").select("id", "label", "features")

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = train.randomSplit(Array(0.7, 0.3), seed = 0)
        // number of iterations
        val numRound = 10
        val numWorkers = 4
        // training parameters
        val paramMap = List(
            "eta" -> 0.023f,
            "max_depth" -> 10,
            "min_child_weight" -> 3.0,
            "subsample" -> 1.0,
            "colsample_bytree" -> 0.82,
            "colsample_bylevel" -> 0.9,
            "base_score" -> 0.005,
            "eval_metric" -> "auc",
            "seed" -> 49,
            "silent" -> 1,
            "objective" -> "binary:logistic").toMap
        println("Starting Xgboost ")
        val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingData, paramMap,round = numRound, nWorkers = numWorkers, useExternalMemory = true)
        val predictions = xgBoostModelWithDF.setExternalMemory(true).transform(testData).select("label", "probabilities")
        // DataFrames can be saved as Parquet files, maintaining the schema information
        predictions.write.save(direct+"preds.parquet")
        //prediction on test set for submission file
        val submission = xgBoostModelWithDF.setExternalMemory(true).transform(test).select("id", "probabilities")
        submission.show(10)
        submission.write.save(direct+"submission.parquet")
        println("done")
        spark.stop()
    }
}
System.exit(0)

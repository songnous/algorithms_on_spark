package org.apache.spark.ml.fpm

import org.apache.spark.ml.conf.SparkJobBase
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{udf,col}
object FPGrowthTest extends SparkJobBase{
  def main(args: Array[String]): Unit = {
    //初始化sparkSession
    implicit val spark: SparkSession = getSparkSession(sparkConf, "WorkItermDataGen")
    spark.sparkContext.setLogLevel("WARN")
    var stockRetail = spark.read
      .format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load("data/stockSeq.csv")
      .select("CodeSeq")
    val splitStr = udf((str:String)=>{str.split(",")})
    stockRetail = stockRetail.withColumn("CodeSeq",splitStr(col("CodeSeq")))

      stockRetail.show(3)
    val fpgrowth = new FPGrowth().
      setMinSupport(0.054).
      setMinConfidence(0.75).setNumPartitions(4).
      setItemsCol("CodeSeq").
      fit(stockRetail)
    fpgrowth.associationRules.show()

    spark.stop()
  }
}

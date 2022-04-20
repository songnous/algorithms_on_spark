package org.apache.spark.ml.dbscan

import org.apache.spark.ml.conf.SparkJobBase
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.functions.{col,udf}

object DBSCANTest extends SparkJobBase {
  def main(args: Array[String]): Unit = {
    //初始化sparkSession
    implicit val spark: SparkSession = getSparkSession(sparkConf, "Spark_DBSCAN_Clustering")
    spark.sparkContext.setLogLevel("WARN")
    /**
     * 读取csv数据
     */
    val origin_data = spark.read
      .format("csv")
      .option("header", false)
      .option("inferSchema", false)
      .load("data/cluto-t7-10k.csv")
//    origin_data.show()
//    val featuring_point = origin_data.select("_c0","_c1").rdd
//      .map{line=>Row(Vectors.dense(Array(line.getAs[String](0).toDouble,line.getAs[String](1).toDouble)))}
//    val df = spark.createDataFrame(featuring_point,StructType(Array(StructField("features",new VectorUDT,false))))
    val pointVector = udf((c0:String,c1:String)=>{
        Vectors.dense(Array(c0.toDouble,c1.toDouble))
    })
    val df = origin_data.select("_c0","_c1").withColumn("features",pointVector(col("_c0"),col("_c1")))
    df.show()
    val dbscan = new DBSCAN2("uid-v0-20220419").setEps(0.3)
      .setMinPoints(10).setMaxPointsPerPartition(1000)
      .fit(df)
    val result = dbscan.transform(df)
    result.show()
  }
}

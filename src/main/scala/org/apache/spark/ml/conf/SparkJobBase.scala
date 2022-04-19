package org.apache.spark.ml.conf

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait SparkJobBase {
  protected lazy val sparkConf: SparkConf = getSparkConf

  def getSparkConf: SparkConf = {
    new SparkConf()
  }

  def getSparkSession(conf: SparkConf, appName: String): SparkSession = {

    val builder = SparkSession.builder()

    builder
      .appName(appName)
      .config(conf)
      .master("local[*]")
//      .enableHiveSupport()
      .getOrCreate()
  }
}

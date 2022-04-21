package org.apache.spark.ml.dbscan

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.dbscan.DBSCANLabeledPoint.Flag
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.{SparkSession}
import org.json4s.{DefaultFormats, JValue}
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.JsonDSL._
import scala.reflect.runtime.universe.TypeTag


/**
  * Top level method for calling DBSCAN
  */
object DBSCAN extends Loader[DBSCAN] {

  /**
   * Train a DBSCAN Model using the given set of parameters
   *
   * @param data                  training points stored as `RDD[Vector]`
   *                              only the first two points of the vector are taken into consideration
   * @param eps                   the maximum distance between two points for them to be considered as part
   *                              of the same region
   * @param minPoints             the minimum number of points required to form a dense region
   * @param maxPointsPerPartition the largest number of points in a single partition
   */
  def train(
             data: RDD[Vector],
             eps: Double,
             minPoints: Int,
             maxPointsPerPartition: Int,
             oldModelPath: String): DBSCAN = {

    new DBSCAN(eps, minPoints, maxPointsPerPartition, oldModelPath, null, null).train(data)

  }

  override def load(sc: SparkContext, path: String): DBSCAN = {
    val (loadedClassName, version, metadata) = Loader.loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName
    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        SaveLoadV1_0.load(sc, path)
      case _ => throw new Exception(
        s"KMeansModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)\n")
    }
  }

  object SaveLoadV1_0 {

      val thisFormatVersion = "1.0"

      val thisClassName = " org.apache.spark.ml.dbscan.DBSCAN"

      def save(sc: SparkContext, model: DBSCAN, path: String): Unit = {
        val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
        val metadata = compact(render(
          ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("eps" -> model.eps)
            ~ ("minPoints" -> model.minPoints) ~ ("maxPointsPerPartition" -> model.maxPointsPerPartition)))
        sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))
        val labeledPoints = model.labeledPoints
        spark.createDataFrame(labeledPoints).write.parquet(Loader.dataPath(path))
      }

      def load(sc: SparkContext, path: String): DBSCAN = {
        implicit val formats = DefaultFormats
        val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
        val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
        assert(className == thisClassName)
        assert(formatVersion == thisFormatVersion)
        val eps = (metadata \ "eps").extract[Double]
        val minPoints = (metadata \ "minPoints").extract[Int]
        val maxPointsPerPartition = (metadata \ "maxPointsPerPartition").extract[Int]
        val labeledPoints = spark.read.parquet(Loader.dataPath(path))
        new DBSCAN(eps, minPoints, maxPointsPerPartition, "", null, labeledPoints.rdd.map(r=>(r.getInt(0),r.get(1).asInstanceOf[DBSCANLabeledPoint])))
      }
  }
}
  /**
   * A parallel implementation of DBSCAN clustering. The implementation will split the data space
   * into a number of partitions, making a best effort to keep the number of points in each
   * partition under `maxPointsPerPartition`. After partitioning, traditional DBSCAN
   * clustering will be run in parallel for each partition and finally the results
   * of each partition will be merged to identify global clusters.
   *
   * This is an iterative algorithm that will make multiple passes over the data,
   * any given RDDs should be cached by the user.
   */
  class DBSCAN private(val eps: Double,
                       val minPoints: Int,
                       val maxPointsPerPartition: Int,
                       val oldModelPath: String,
                       @transient val partitions: List[(Int, DBSCANRectangle)],
                       private val labeledPartitionedPoints: RDD[(Int, DBSCANLabeledPoint)])
    extends Serializable with Saveable with Logging {

    type Margins = (DBSCANRectangle, DBSCANRectangle, DBSCANRectangle)
    type ClusterId = (Int, Int)

    def minimumRectangleSize: Double = 2 * eps

    def labeledPoints: RDD[DBSCANLabeledPoint] = {
      labeledPartitionedPoints.values
    }

    private def train(vectors: RDD[Vector]): DBSCAN = {
      // generate the smallest rectangles that split the space
      // and count how many points are contained in each one of them
      val minimumRectanglesWithCount =
      vectors
        .map(toMinimumBoundingRectangle)
        .map((_, 1))
        .aggregateByKey(0)(_ + _, _ + _)
        .collect()
        .toSet

      // find the best partitions for the data space
      val localPartitions = EvenSplitPartitioner
        .partition(minimumRectanglesWithCount, maxPointsPerPartition, minimumRectangleSize)

      logDebug("Found partitions: ")
      localPartitions.foreach(p => logDebug(p.toString))

      // grow partitions to include eps
      val localMargins =
        localPartitions
          .map({ case (p, _) => (p.shrink(eps), p, p.shrink(-eps)) })
          .zipWithIndex

      val margins = vectors.context.broadcast(localMargins)

      // assign each point to its proper partition
      val duplicated = for {
        point <- vectors.map(DBSCANPoint)
        ((inner, main, outer), id) <- margins.value
        if outer.contains(point)
      } yield (id, point)

      val numOfPartitions = localPartitions.size

      // perform local dbscan
      val clustered =
        duplicated
          .groupByKey(numOfPartitions)
          .flatMapValues(points =>
            new LocalDBSCANNaive(eps, minPoints).fit(points))
          .cache()

      // find all candidate points for merging clusters and group them
      val mergePoints =
        clustered
          .flatMap({
            case (partition, point) =>
              margins.value
                .filter({
                  case ((inner, main, _), _) => main.contains(point) && !inner.almostContains(point)
                })
                .map({
                  case (_, newPartition) => (newPartition, (partition, point))
                })
          })
          .groupByKey()

      logDebug("About to find adjacencies")
      // find all clusters with aliases from merging candidates
      val adjacencies =
        mergePoints
          .flatMapValues(findAdjacencies)
          .values
          .collect()

      // generated adjacency graph
      val adjacencyGraph = adjacencies.foldLeft(DBSCANGraph[ClusterId]()) {
        case (graph, (from, to)) => graph.connect(from, to)
      }

      logDebug("About to find all cluster ids")
      // find all cluster ids
      val localClusterIds =
        clustered
          .filter({ case (_, point) => point.flag != Flag.Noise })
          .mapValues(_.cluster)
          .distinct()
          .collect()
          .toList

      // assign a global Id to all clusters, where connected clusters get the same id
      val (total, clusterIdToGlobalId) = localClusterIds.foldLeft((0, Map[ClusterId, Int]())) {
        case ((id, map), clusterId) => {

          map.get(clusterId) match {
            case None => {
              val nextId = id + 1
              val connectedClusters = adjacencyGraph.getConnected(clusterId) + clusterId
              logDebug(s"Connected clusters $connectedClusters")
              val toadd = connectedClusters.map((_, nextId)).toMap
              (nextId, map ++ toadd)
            }
            case Some(x) =>
              (id, map)
          }

        }
      }

      logDebug("Global Clusters")
      clusterIdToGlobalId.foreach(e => logDebug(e.toString))
      logInfo(s"Total Clusters: ${localClusterIds.size}, Unique: $total")

      val clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

      logDebug("About to relabel inner points")
      // relabel non-duplicated points
      val labeledInner =
        clustered
          .filter(isInnerPoint(_, margins.value))
          .map {
            case (partition, point) => {

              if (point.flag != Flag.Noise) {
                point.cluster = clusterIds.value((partition, point.cluster))
              }

              (partition, point)
            }
          }

      logDebug("About to relabel outer points")
      // de-duplicate and label merge points
      val labeledOuter =
        mergePoints.flatMapValues(partition => {
          partition.foldLeft(Map[DBSCANPoint, DBSCANLabeledPoint]())({
            case (all, (partition, point)) =>

              if (point.flag != Flag.Noise) {
                point.cluster = clusterIds.value((partition, point.cluster))
              }

              all.get(point) match {
                case None => all + (point -> point)
                case Some(prev) => {
                  // override previous entry unless new entry is noise
                  if (point.flag != Flag.Noise) {
                    prev.flag = point.flag
                    prev.cluster = point.cluster
                  }
                  all
                }
              }

          }).values
        })

      val finalPartitions = localMargins.map {
        case ((_, p, _), index) => (index, p)
      }
      logDebug("Done")
      new DBSCAN(
        eps,
        minPoints,
        maxPointsPerPartition,
        oldModelPath,
        finalPartitions,
        labeledInner.union(labeledOuter))

    }

    /**
     * increment training
     * 步骤：
     * 1.加载上一版本DBSCAN得训练模型，去除所有聚类结果得中心点，即label_point 得Flag 为Core
     * 2.计算新来得一批增量数据得每个点与 已有聚类中心点得距离：得到距离最近的Core点信息 及 min_distance
     * 3.检查这个新的点
     * 一、周围的点数量 num_ps > minPoints  && min_distance < new eps
     * 将这个新的点归类到这个聚簇当中，并且更新这个聚簇的中心
     * 二、如果与已有最近中心的距离 min_distance > new eps ，则这个点不属于已有类 ，暂且标记为noise 放在一个数据结构里 beyond_points
     * 对这个beyond_points 重新进行localDBSCAN算法，看聚类成新的簇信息， 最后与已有模型合并(已有模型包括 标记点的信息更新、中心点更新、
     * 点都为已访问，当然参数minPoints，eps都更新为新的。说明：一般增量的话，按理说minPoints、eps
     * 这种参数不应改变，而是对整体数据重新训练，视业务情况而决定)
     *
     */
    private def train(vectors: RDD[Vector], oldModelPath: String): Unit = {
      //加载oldmodel


      // generate the smallest rectangles that split the space
      // and count how many points are contained in each one of them
      //    val minimumRectanglesWithCount =
      //    vectors
      //      .map(toMinimumBoundingRectangle)
      //      .map((_, 1))
      //      .aggregateByKey(0)(_ + _, _ + _)
      //      .collect()
      //      .toSet
      //
      //    // find the best partitions for the data space
      //    val localPartitions = EvenSplitPartitioner
      //      .partition(minimumRectanglesWithCount, maxPointsPerPartition, minimumRectangleSize)
    }

    /**
     * Find the appropriate label to the given `vector`
     *
     * This method is not yet implemented
     */
    def predict(vector: Vector): Double = {
      var centerid = 0
      partitions.foreach { x =>
        if (x._2.contains(DBSCANPoint(vector))) {
          centerid = x._1
        }
      }
      centerid.toDouble
    }

    private def isInnerPoint(
                              entry: (Int, DBSCANLabeledPoint),
                              margins: List[(Margins, Int)]): Boolean = {
      entry match {
        case (partition, point) =>
          val ((inner, _, _), _) = margins.filter({
            case (_, id) => id == partition
          }).head

          inner.almostContains(point)
      }
    }

    private def findAdjacencies(partition: Iterable[(Int, DBSCANLabeledPoint)]):
    Set[((Int, Int), (Int, Int))] = {

      val zero = (Map[DBSCANPoint, ClusterId](), Set[(ClusterId, ClusterId)]())

      val (seen, adjacencies) = partition.foldLeft(zero)({
        case ((seen, adjacencies), (partition, point)) =>
          // noise points are not relevant for adjacencies
          if (point.flag == Flag.Noise) {
            (seen, adjacencies)
          } else {
            val clusterId = (partition, point.cluster)
            seen.get(point) match {
              case None => (seen + (point -> clusterId), adjacencies)
              case Some(prevClusterId) => (seen, adjacencies + ((prevClusterId, clusterId)))
            }

          }
      })

      adjacencies
    }

    private def toMinimumBoundingRectangle(vector: Vector): DBSCANRectangle = {
      val point = DBSCANPoint(vector)
      val x = corner(point.x)
      val y = corner(point.y)
      DBSCANRectangle(x, y, x + minimumRectangleSize, y + minimumRectangleSize)
    }

    private def corner(p: Double): Double =
      (shiftIfNegative(p) / minimumRectangleSize).intValue * minimumRectangleSize

    private def shiftIfNegative(p: Double): Double =
      if (p < 0) p - minimumRectangleSize else p


    /**
     * 保存模型方法
     *
     * @param sc
     * @param path
     */
    override def save(sc: SparkContext, path: String): Unit = {
      DBSCAN.SaveLoadV1_0.save(sc, this, path)
    }

    override protected def formatVersion: String = "1.0"
}


/**
 * 复制mllib Loader
 */
private[ml] object Loader {

  /** Returns URI for path/data using the Hadoop filesystem */
  def dataPath(path: String): String = new Path(path, "data").toUri.toString

  /** Returns URI for path/metadata using the Hadoop filesystem */
  def metadataPath(path: String): String = new Path(path, "metadata").toUri.toString

  /**
   * Check the schema of loaded model data.
   *
   * This checks every field in the expected schema to make sure that a field with the same
   * name and DataType appears in the loaded schema.  Note that this does NOT check metadata
   * or containsNull.
   *
   * @param loadedSchema  Schema for model data loaded from file.
   * @tparam Data  Expected data type from which an expected schema can be derived.
   */
  def checkSchema[Data: TypeTag](loadedSchema: StructType): Unit = {
    // Check schema explicitly since erasure makes it hard to use match-case for checking.
    val expectedFields: Array[StructField] =
      ScalaReflection.schemaFor[Data].dataType.asInstanceOf[StructType].fields
    val loadedFields: Map[String, DataType] =
      loadedSchema.map(field => field.name -> field.dataType).toMap
    expectedFields.foreach { field =>
      assert(loadedFields.contains(field.name), s"Unable to parse model data." +
        s"  Expected field with name ${field.name} was missing in loaded schema:" +
        s" ${loadedFields.mkString(", ")}")
      assert(loadedFields(field.name).sameType(field.dataType),
        s"Unable to parse model data.  Expected field $field but found field" +
          s" with different type: ${loadedFields(field.name)}")
    }
  }

  /**
   * Load metadata from the given path.
   * @return (class name, version, metadata)
   */
  def loadMetadata(sc: SparkContext, path: String): (String, String, JValue) = {
    implicit val formats = DefaultFormats
    val metadata = parse(sc.textFile(metadataPath(path)).first())
    val clazz = (metadata \ "class").extract[String]
    val version = (metadata \ "version").extract[String]
    (clazz, version, metadata)
  }
}





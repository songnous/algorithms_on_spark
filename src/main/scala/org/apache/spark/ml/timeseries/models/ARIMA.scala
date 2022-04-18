package org.apache.spark.ml.timeseries.models

import breeze.linalg.{diag, sum, DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector}
import org.apache.commons.math3.analysis.{MultivariateFunction, MultivariateVectorFunction}
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar.{GoalType, ObjectiveFunction, ObjectiveFunctionGradient}
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.timeseries.{Lag, MatrixUtil, UnivariateTimeSeries}
import org.apache.spark.ml.timeseries.params.TimeSeriesParams
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
  * ARIMA models allow modeling timeseries as a function of prior values of the series
  * (i.e. autoregressive terms) and a moving average of prior error terms. ARIMA models
  * are traditionally specified as ARIMA(p, d, q), where p is the autoregressive order,
  * d is the differencing order, and q is the moving average order. Using the backshift (aka
  * lag operator) B, which when applied to a Y returns the prior value, the
  * ARIMA model can be specified as
  * Y_t = c + \sum_{i=1}^p \phi_i*B^i*Y_t + \sum_{i=1}^q \theta_i*B^i*\epsilon_t + \epsilon_t
  * where Y_i has been differenced as appropriate according to order `d`
  * See [[https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average]] for more
  * information on ARIMA.
  * See [[https://en.wikipedia.org/wiki/Order_of_integration]] for more information on differencing
  * integrated time series.
  */

trait ARIMAModelParams extends TimeSeriesParams {
  final val nFuture = new Param[Int](this, "nFuture", "Periods in the future to forecast")
  def setNFuture(value: Int): this.type = set(nFuture, value)
}

trait ARIMAParams extends ARIMAModelParams{

  final val maxEval = new Param[Int](this, "maxEval", "max eval")
  def setMaxEval(value: Int): this.type = set(maxEval, value)

  final val maxIter = new Param[Int](this, "maxIter", "max iteration")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  final val p = new Param[Int](this, "p", "autoregressive order")
  def setP(value: Int): this.type = set(p, value)

  final val d = new Param[Int](this, "d", "differencing order")
  def setD(value: Int): this.type = set(d, value)

  final val q = new Param[Int](this, "q", "moving average order")
  def setQ(value: Int): this.type = set(q, value)

  final val includeIntercept = new Param[Boolean](this, "includeIntercept",
    "Add intercept term")
  def setIncludeIntercept(value: Boolean): this.type = set(includeIntercept, value)

  final val method = new Param[String](this, "method", "object function and" +
    " optimization method, current options are 'ccs-bobyqa' and 'ccs-cgd'.Default is 'ccs-cgd'")
  def setMethod(value: String): this.type = set(method, value)

  final val initParams = new Param[Array[Double]](this, "initParams",
      "A set of user provided initial parameters for optimization. if null(default)," +
      " initialized using Hunnan-Rissanen algorithm. If provided, order of parameter" +
      " should be: intercept term, AR parameters, MA parameters.")
  def setInitParams(value: Array[Double]): this.type = set(initParams, value)

}

/**
  * Given a time series, fit a non-seasonal ARIMA model of order (p, d, q), where p represents
  * the autoregression terms, d represents the order of differencing, and q moving average error
  * terms. If includeIntercept is true, the model is fitted with an intercept. In order to select
  * the appropriate order of the model, users are advised to inspect ACF and PACF plots, or compare
  * the values of the objective function. Finally, while the current implementation of
  * `fitModel` verifies that parameters fit stationarity and invertibility requirements,
  * there is currently no function to transform them if they do not. It is up to the user
  * to make these changes as appropriate (or select a different model specification)
  */

class ARIMA(override val uid: String) extends Estimator[ARIMAModel] with ARIMAParams{

    setDefault(includeIntercept -> true,
      method -> "css-cgd",
      initParams -> null,
      timeCol -> "time",
      timeSeriesCol -> "timeseries",
      maxIter -> 10000,
      maxEval -> 10000)

  def this() = this(Identifiable.randomUID("ARIMA"))
  /**
    * Fits a model to the input data.
    */
  override def fit(dataset: Dataset[_]): ARIMAModel = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()

    val dataVector = Vectors.dense(data.map(x => x._2))

    // Drop first d terms, can't use those to fit, since not at the right level of differencing
    val diffedTs = UnivariateTimeSeries.differencesOfOrderD(dataVector, ${d}).toArray.drop(${d})

    val dataFrame = generateDf(dataset.sparkSession, diffedTs)

    if (${initParams} == null) setInitParams(hannanRissanenInit(dataFrame, diffedTs))

    val params = ${method} match {
      case "css-bobyqa" => fitWithCSSBOBYQA(diffedTs)
      case "css-cgd" => fitWithCSSCGD(diffedTs)
      case _ => throw new UnsupportedOperationException()
    }

    val model = new ARIMAModel(${p}, ${d}, ${q}, params, ${includeIntercept})
      .setTimeCol(${timeCol})
      .setTimeSeriesCol(${timeSeriesCol})

    model
  }

  private def generateDf(sparkSession: SparkSession, array: Array[Double]): DataFrame = {
    val schema = StructType(Array(StructField(${timeCol}, StringType), StructField(${timeSeriesCol},
      DoubleType)))

    val rdd = sparkSession.sparkContext.parallelize(
      array.zipWithIndex.map(x => Row(x._2.formatted("%010d"), x._1)))

    sparkSession.createDataFrame(rdd, schema)
  }

  /**
    * Initializes ARMA parameter estimates using the Hannan-Rissanen algorithm. The process is:
    * fit an AR(m) model of a higher order (i.e. m > max(p, q)), use this to estimate errors,
    * then fit an OLS model of AR(p) terms and MA(q) terms. The coefficients estimated by
    * the OLS model are returned as initial parameter estimates.
    * See [[http://halweb.uc3m.es/esp/Personal/personas/amalonso/esp/TSAtema9.pdf]] for more
    * information.

    * @param y time series to be modeled
    * @return initial ARMA(p, d, q) parameter estimates
    */
 def hannanRissanenInit(dataFrame: DataFrame, y: Array[Double]): Array[Double] = {
    val addToLag = 1
    // m > max(p, q) for higher order requirement
    val m = math.max(${p}, ${q}) + addToLag

    // higher order AR(m) model
    val arModel = new Autoregression()
        .setTimeCol(${timeCol})
        .setTimeSeriesCol(${timeSeriesCol})
        .setMaxLag(m)
        .fit(dataFrame)

    val arTerms1 = Lag.lagMatTrimBoth(y, m, includeOriginal = false)
    val yTrunc = y.drop(m)

    val estimated = arTerms1.zip(
      Array.fill(yTrunc.length)(arModel.coefficients)
    ).map { case (v, b) => v.zip(b).map { case (yi, bi) => yi * bi }.sum + arModel.c }
    // errors estimated from AR(m)
    val errors = yTrunc.zip(estimated).map { case (yi, yhat) => yi - yhat }

    // secondary regression, regresses X_t on AR and MA terms
    val arTerms2 = Lag.lagMatTrimBoth(yTrunc, ${p}, includeOriginal = false)
      .drop(math.max(${q} - ${p}, 0))

    val errorTerms = Lag.lagMatTrimBoth(errors, ${q}, includeOriginal = false)
      .drop(math.max(${p} - ${q}, 0))

    val allTerms = arTerms2.zip(errorTerms).map { case (ar, ma) => ar ++ ma }
    val regression = new OLSMultipleLinearRegression()
    regression.setNoIntercept(!${includeIntercept})
    regression.newSampleData(yTrunc.drop(m - addToLag), allTerms)
    val params = regression.estimateRegressionParameters()
    params
  }

  /**
    * Fit an ARIMA model using conditional sum of squares estimator, optimized using unbounded
    * BOBYQA.
    * @param diffedY differenced time series, as appropriate
    * @return parameters optimized using CSS estimator, with method BOBYQA
    */
  private def fitWithCSSBOBYQA(diffedY: Array[Double]): Array[Double] = {
    // We set up starting/ending trust radius using default suggested in
    // http://cran.r-project.org/web/packages/minqa/minqa.pdf
    // While # of interpolation points as mentioned common in
    // http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf
    val radiusStart = math.min(0.96, 0.2 * ${initParams}.map(math.abs).max)
    val radiusEnd = radiusStart * 1e-6
    val dimension = ${p} + ${q} + (if (${includeIntercept}) 1 else 0)
    val interpPoints = dimension * 2 + 1

    val optimizer = new BOBYQAOptimizer(interpPoints, radiusStart, radiusEnd)
    val objFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new ARIMAModel(${p}, ${d}, ${q}, params, ${includeIntercept}).logLikelihoodCSSARMA(diffedY)
      }
    })

    val initialGuess = new InitialGuess(${initParams})
    val bounds = SimpleBounds.unbounded(dimension)
    val goal = GoalType.MAXIMIZE
    val optimal = optimizer.optimize(objFunction, goal, bounds, new MaxIter(${maxIter}),
      new MaxEval(${maxEval}), initialGuess)
    optimal.getPoint
  }

  /**
    * Fit an ARIMA model using conditional sum of squares estimator, optimized using conjugate
    * gradient descent
    * @param diffedY differenced time series, as appropriate
    * @return parameters optimized using CSS estimator, with method conjugate gradient descent
    */
  private def fitWithCSSCGD(diffedY: Array[Double]): Array[Double] = {

    val optimizer = new NonLinearConjugateGradientOptimizer(
      NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES,
      new SimpleValueChecker(1e-7, 1e-7))
    val objFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new ARIMAModel(${p}, ${d}, ${q}, params, ${includeIntercept}).logLikelihoodCSSARMA(diffedY)
      }
    })
    val gradient = new ObjectiveFunctionGradient(new MultivariateVectorFunction() {
      def value(params: Array[Double]): Array[Double] = {
        new ARIMAModel(${p}, ${d}, ${q}, params, ${includeIntercept})
          .gradientlogLikelihoodCSSARMA(diffedY)
      }
    })
    val initialGuess = new InitialGuess(${initParams})
    val maxIter = new MaxIter(10000)
    val maxEval = new MaxEval(10000)
    val goal = GoalType.MAXIMIZE
    val optimal = optimizer.optimize(objFunction, gradient, goal, initialGuess, maxIter, maxEval)
    optimal.getPoint
  }


  override def copy(extra: ParamMap): Estimator[ARIMAModel] = defaultCopy(extra)

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    schema
  }
}

class ARIMAModel(override val uid: String, val p: Int, val d: Int, val q: Int,
      val coefficient: Array[Double], val hasIntercept: Boolean = true) extends
  Model[ARIMAModel] with ARIMAModelParams{

  setDefault(timeCol -> "time",
    timeSeriesCol -> "timeseries")

  override def copy(extra: ParamMap): ARIMAModel = defaultCopy(extra)

  def this(p: Int, d: Int, q: Int, coefficient: Array[Double], hasIntercept: Boolean ) =
    this(Identifiable.randomUID("ARIMAModel"), p, d, q, coefficient, hasIntercept)

  def this(p: Int, d: Int, q: Int, coefficient: Array[Double]) =
    this(Identifiable.randomUID("ARIMAModel"), p, d, q, coefficient)
  /**
    * log likelihood based on conditional sum of squares
    *
    * Source: http://www.nuffield.ox.ac.uk/economics/papers/1997/w6/ma.pdf
    * @return log likelihood
    */
  def logLikelihoodCSS(y: Vector): Double = {
    val diffedY = UnivariateTimeSeries.differencesOfOrderD(y, d).toArray.drop(d)
    logLikelihoodCSSARMA(diffedY)
  }

  /**
    * log likelihood based on conditional sum of squares. In contrast to logLikelihoodCSS the array
    * provided should correspond to an already differenced array, so that the function below
    * corresponds to the log likelihood for the ARMA rather than the ARIMA process
    *
    * @param diffedY differenced array
    * @return log likelihood of ARMA
    */
  def logLikelihoodCSSARMA(diffedY: Array[Double]): Double = {
    val n = diffedY.length
    val yHat = new BreezeDenseVector(Array.fill(n)(0.0))
    val yVec = new DenseVector(diffedY)

    iterateARMA(yVec, yHat, _ + _, goldStandard = yVec)

    val maxLag = math.max(p, q)
    // drop first maxLag terms, since we can't estimate residuals there, since no
    // AR(n) terms available
    val css = diffedY.zip(yHat.toArray).drop(maxLag).map { case (obs, pred) =>
      math.pow(obs - pred, 2)
    }.sum
    val sigma2 = css / n
    (-n / 2) * math.log(2 * math.Pi * sigma2) - css / (2 * sigma2)
  }

  /**
    * Perform operations with the AR and MA terms, based on the time series `ts` and the errors
    * based off of `goldStandard` or `errors`, combined with elements from the series `dest`.
    * Weights for terms are taken from the current model configuration.
    * So for example: iterateARMA(series1, series_of_zeros,  _ + _ , goldStandard = series1,
    * initErrors = null)
    * calculates the 1-step ahead ARMA forecasts for series1 assuming current coefficients, and
    * initial MA errors of 0.
    *
    * @param ts Time series to use for AR terms
    * @param dest Time series holding initial values at each index
    * @param op Operation to perform between values in dest, and various combinations of ts, errors
    *           and intercept terms
    * @param goldStandard The time series to which to compare 1-step ahead forecasts to obtain
    *                     moving average errors. Default set to null. In which case, we check if
    *                     errors are directly provided. Either goldStandard or errors can be null,
    *                     but not both
    * @param errors The time series of errors to be used as error at time i, which is then used
    *               for moving average terms in future indices. Either goldStandard or errors can
    *               be null, but not both
    * @param initMATerms Initialization for first q terms of moving average. If none provided (i.e.
    *                    remains null, as per default), then initializes to all zeros
    * @return the time series resulting from the interaction of the parameters with the model's
    *         coefficients
    */
  private def iterateARMA(ts: Vector, dest: BreezeDenseVector[Double],
              op: (Double, Double) => Double, goldStandard: Vector = null,
              errors: Vector = null, initMATerms: Array[Double] = null):
        BreezeDenseVector[Double] = {

    require(goldStandard != null || errors != null, "goldStandard or errors must be passed in")
    val maTerms = if (initMATerms == null) Array.fill(q)(0.0) else initMATerms
    val intercept = if (hasIntercept) 1 else 0
    // maximum lag
    var i = math.max(p, q)
    var j = 0
    val n = ts.size
    var error = 0.0

    while (i < n) {
      j = 0
      // intercept
      dest(i) = op(dest(i), intercept * coefficient(j))
      // autoregressive terms
      while (j < p && i - j - 1 >= 0) {
        dest(i) = op(dest(i), ts(i - j - 1) * coefficient(intercept + j))
        j += 1
      }
      // moving average terms
      j = 0
      while (j < q) {
        dest(i) = op(dest(i), maTerms(j) * coefficient(intercept + p + j))
        j += 1
      }

      error = if (goldStandard == null) errors(i) else goldStandard(i) - dest(i)
      updateMAErrors(maTerms, error)
      i += 1
    }
    dest
  }

  /**
    * Updates the error vector in place with a new (more recent) error
    * The newest error is placed in position 0, while older errors "fall off the end"
    *
    * @param errs array of errors of length q in ARIMA(p, d, q), holds errors for t-1 through t-q
    * @param newError the error at time t
    * @return a modified array with the latest error placed into index 0
    */
  private def updateMAErrors(errs: Array[Double], newError: Double): Unit = {
    val n = errs.length
    var i = 0
    while (i < n - 1) {
      errs(i + 1) = errs(i)
      i += 1
    }
    if (n > 0) {
      errs(0) = newError
    }
  }

  /**
    * Calculates the gradient for the log likelihood function using CSS
    * Derivation:
    * L(y | \theta) = -\frac{n}{2}log(2\pi\sigma^2) - \frac{1}{2\pi}\sum_{i=1}^n \epsilon_t^2 \\
    * \sigma^2 = \frac{\sum_{i = 1}^n \epsilon_t^2}{n} \\
    * \frac{\partial L}{\partial \theta} = -\frac{1}{\sigma^2}
    * \sum_{i = 1}^n \epsilon_t \frac{\partial \epsilon_t}{\partial \theta} \\
    * \frac{\partial \epsilon_t}{\partial \theta} = -\frac{\partial \hat{y}}{\partial \theta} \\
    * \frac{\partial\hat{y}}{\partial c} = 1 +
    * \phi_{t-q}^{t-1}*\frac{\partial \epsilon_{t-q}^{t-1}}{\partial c} \\
    * \frac{\partial\hat{y}}{\partial \theta_{ar_i}} =  y_{t - i} +
    * \phi_{t-q}^{t-1}*\frac{\partial \epsilon_{t-q}^{t-1}}{\partial \theta_{ar_i}} \\
    * \frac{\partial\hat{y}}{\partial \theta_{ma_i}} =  \epsilon_{t - i} +
    * \phi_{t-q}^{t-1}*\frac{\partial \epsilon_{t-q}^{t-1}}{\partial \theta_{ma_i}} \\
    *
    * @param diffedY array of differenced values
    * @return
    */
  def gradientlogLikelihoodCSSARMA(diffedY: Array[Double]): Array[Double] = {
    val n = diffedY.length
    // fitted
    val yHat = new BreezeDenseVector[Double](Array.fill(n)(0.0))
    // reference values (i.e. gold standard)
    val yRef = new DenseVector(diffedY)
    val maxLag = math.max(p, q)
    val intercept = if (hasIntercept) 1 else 0
    val maTerms = Array.fill[Double](q)(0.0)

    // matrix of error derivatives at time t - 0, t - 1, ... t - q
    val dEdTheta = new BreezeDenseMatrix[Double](q + 1, coefficient.length)
    val gradient = new BreezeDenseVector[Double](Array.fill(coefficient.length)(0.0))

    // error-related
    var error = 0.0
    var sigma2 = 0.0

    // iteration-related
    var i = maxLag
    var j = 0
    var k = 0

    while (i < n) {
      j = 0
      // initialize partial error derivatives in each iteration to weighted average of
      // prior partial error derivatives, using moving average coefficients
      while (j < coefficient.length) {
        k = 0
        while (k < q) {
          dEdTheta(0, j) -= coefficient(intercept + p + k) * dEdTheta(k + 1, j)
          k += 1
        }
        j += 1
      }
      // intercept
      j = 0
      yHat(i) += intercept * coefficient(j)
      dEdTheta(0, 0) -= intercept

      // autoregressive terms
      while (j < p && i - j - 1 >= 0) {
        yHat(i) += yRef(i - j - 1) * coefficient(intercept + j)
        dEdTheta(0, intercept + j) -= yRef(i - j - 1)
        j += 1
      }

      // moving average terms
      j = 0
      while (j < q) {
        yHat(i) += maTerms(j) * coefficient(intercept + p + j)
        dEdTheta(0, intercept + p + j) -= maTerms(j)
        j += 1
      }

      error = yRef(i) - yHat(i)
      sigma2 += math.pow(error, 2) / n
      updateMAErrors(maTerms, error)
      // update gradient
      gradient :+= (dEdTheta(0, ::) * error).t
      // shift back error derivatives to make room for next period
      dEdTheta(1 to -1, ::) := dEdTheta(0 to -2, ::)
      // reset latest partial error derivatives to 0
      dEdTheta(0, ::) := 0.0
      i += 1
    }

    gradient := gradient :/ -sigma2
    gradient.toArray
  }

  /**
    * Provided fitted values for timeseries ts as 1-step ahead forecasts, based on current
    * model parameters, and then provide `nFuture` periods of forecast. We assume AR terms
    * prior to the start of the series are equal to the model's intercept term (or 0.0, if fit
    * without and intercept term).Meanwhile, MA terms prior to the start are assumed to be 0.0. If
    * there is differencing, the first d terms come from the original series.
    *
    * @param dataset Timeseries to use as gold-standard. Each value (i) in the returning series
    *           is a 1-step ahead forecast of ts(i). We use the difference between ts(i) -
    *           estimate(i) to calculate the error at time i, which is used for the moving
    *           average terms.
    * @return a series consisting of fitted 1-step ahead forecasts for historicals and then
    *         `nFuture` periods of forecasts. Note that in the future values error terms become
    *         zero and prior predictions are used for any AR terms.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()

    val dataVector = Vectors.dense(data.map(x => x._2))

    val maxLag = math.max(p, q)

    // difference timeseries as necessary for model
    val diffedTs = new DenseVector(UnivariateTimeSeries.differencesOfOrderD(dataVector, d)
      .toArray.drop(d))

    // Assumes prior AR terms are equal to model intercept
    val interceptAmt = if (hasIntercept) coefficient(0) else 0.0
    val diffedTsExtended = new DenseVector(Array.fill(maxLag)(interceptAmt) ++ diffedTs.toArray)
    val histLen = diffedTsExtended.size
    val hist = new BreezeDenseVector[Double](Array.fill(histLen)(0.0))

    // fit historical values (really differences in case of d > 0)
    iterateARMA(diffedTsExtended, hist, _ + _, goldStandard = diffedTsExtended)

    // Last set of errors, to be used in forecast if MA terms included
    val maTerms = (for (i <- histLen - maxLag until histLen) yield {
      diffedTsExtended(i) - hist(i)
    }).toArray

    // copy over last maxLag values, to use in iterateARMA for forward curve
    val forward = new BreezeDenseVector[Double](Array.fill(${nFuture} + maxLag)(0.0))
    if (maxLag > 0) forward(0 until maxLag) := hist(-maxLag to -1)
    // use self as ts to take AR from same series, use self as goldStandard to induce future errors
    // of zero, and use prior moving average errors as initial error terms for MA
    iterateARMA(MatrixUtil.fromBreeze(forward), forward, _ + _,
      goldStandard = MatrixUtil.fromBreeze(forward), initMATerms = maTerms)

    val results = new BreezeDenseVector[Double](Array.fill(dataVector.size + ${nFuture})(0.0))
    // copy over first d terms, since we have no corresponding differences for these
    results(0 until d) := MatrixUtil.toBreeze(dataVector)(0 until d)
    // copy over historicals, drop first maxLag terms (our assumed AR terms)
    results(d until d + histLen - maxLag) := hist(maxLag to -1)
    // drop first maxLag terms from forward curve before copying, these are part of hist already
    results(dataVector.size to -1) := forward(maxLag to -1)

    if (d != 0) {
      // we need to create 1-step ahead forecasts for the integrated series for fitted values
      // by backing through the d-order differences
      // create and fill a matrix of the changes of order i = 0 through d
      val diffMatrix = new BreezeDenseMatrix[Double](d + 1, dataVector.size)
      diffMatrix(0, ::) := MatrixUtil.toBreeze(dataVector).t

      for (i <- 1 to d) {
        // create incremental differences of each order
        // recall that differencesOfOrderD skips first `order` terms, so make sure
        // to advance start as appropriate
        val diffVec = diffMatrix(i - 1, i to -1).t
        diffMatrix(i, i to -1) := MatrixUtil.toBreeze(
          UnivariateTimeSeries.differencesOfOrderD(MatrixUtil.fromBreeze(diffVec), 1)).t
      }

      // historical values are done as 1 step ahead forecasts
      for (i <- d until histLen - maxLag) {
        // Add the fitted change value to elements at time i - 1 and of difference order < d
        // e.g. if d = 2, then we modeled value at i = 3 as (y_3 - y_2) - (y_2 - y_1), so we add
        // to it y_2 - y_1 (which is at diffMatrix(1, 2)) and y_2 (which is at diffMatrix(0, 2) to
        // get an estimate for y_3
        results(i) = sum(diffMatrix(0 until d, i - 1)) + hist(maxLag + i)
      }

      // Take diagonal of last d terms, of order < d,
      // so that we can inverse differencing appropriately with future values
      val prevTermsForForwardInverse = diag(diffMatrix(0 until d, -d to -1))
      // for forecasts, drop first maxLag terms
      val vec = BreezeDenseVector.vertcat(prevTermsForForwardInverse, forward(maxLag to -1))
      val forwardIntegrated = UnivariateTimeSeries.inverseDifferencesOfOrderD(
        MatrixUtil.fromBreeze(vec), d
      )
      // copy into results
      results(-(d + ${nFuture}) to -1) := MatrixUtil.toBreeze(forwardIntegrated)
    }

    val resRDD = dataset.sparkSession.sparkContext.parallelize(results.toArray.map(x => Row(x)))

    val structType = transformSchema(dataset.schema)

    dataset.sparkSession.createDataFrame(resRDD, structType)
  }

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    StructType(Array(StructField("ARIMA", DoubleType)))
  }

  /**
    * Given a timeseries, apply an ARIMA(p, d, q) model to it.
    * We assume that prior MA terms are 0.0 and prior AR terms are equal to the intercept or 0.0 if
    * fit without an intercept
    *
    * @param ts Time series of i.i.d. observations.
    * @return The dest series, representing the application of the model to provided error
    *         terms, for convenience.
    */
  def addTimeDependentEffects(ts: Vector, destTs: Vector): Vector = {
    val maxLag = math.max(p, q)
    val interceptAmt = if (hasIntercept) coefficient(0) else 0.0
    // extend vector so that initial AR(maxLag) are equal to intercept value
    // Recall iterateARMA begins at index maxLag, and uses previous values as AR terms
    val changes = new BreezeDenseVector[Double](Array.fill(maxLag)(interceptAmt) ++ ts.toArray)
    // ts corresponded to errors, and we want to use them
    val errorsProvided = changes.copy
    iterateARMA(MatrixUtil.fromBreeze(changes),
      changes, _ + _, errors = MatrixUtil.fromBreeze(errorsProvided))
    // drop assumed AR terms at start and perform any inverse differencing required
    MatrixUtil.toBreeze(destTs) := MatrixUtil.toBreeze(UnivariateTimeSeries
      .inverseDifferencesOfOrderD(MatrixUtil.fromBreeze(changes(maxLag to -1)), d))
    destTs
  }

  /**
    * Calculates an approximation to the Akaike Information Criterion (AIC).
    * This is an approximation as we use the conditional likelihood, rather than the exact
    * likelihood. Please see [[https://en.wikipedia.org/wiki/Akaike_information_criterion]] for
    * more information on this measure.
    *
    * @param ts the timeseries to evaluate under current model
    * @return an approximation to the AIC under the current model
    */
  def approxAIC(ts: Vector): Double = {
    val conditionalLogLikelihood = logLikelihoodCSS(ts)
    val interceptTerm = if (hasIntercept) 1 else 0
    -2 * conditionalLogLikelihood + 2 * (p + q + interceptTerm)
  }

}

package lks9;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;

import static lks9.varReferences.*;
import org.apache.hadoop.fs.s3a.Constants.*;

public class TrainingModel {

 /**
  * The main method of the application.
  * @param args The command line arguments.
  */
	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
		Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);
		Logger.getLogger("com.github").setLevel(Level.ERROR);
		//error check logger
		SparkSession sprk1 = SparkSession.builder().appName(APP_NAME).master("local[*]")
				.config("spark.executor.memory", "2147480000").config("spark.driver.memory", "2147480000")
				.config("spark.testing.memory", "2147480000")

				.getOrCreate();
				//ignore space above

		if (StringUtils.isNotEmpty(ACCESS_KEY_ID) && StringUtils.isNotEmpty(SECRET_KEY)) {
			sprk1.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
			sprk1.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);
		}
		if (new File(TRAINING_DATASET).exists())
			(new TrainingModel()).logRegMain(sprk1);
		else
			System.out.println("TrainingDataSet.csv doesn't exist");
	}

 /**
  * This function is used to train the model.
  * @param s The spark session.
  */
	public void logRegMain(SparkSession s) {
		System.out.println();
		// don't change above or formatting will break
		Dataset<Row> lblFdf = getDataFrame(s, true, TRAINING_DATASET).cache();
		LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0);
		Pipeline pL = new Pipeline();
		pL.setStages(new PipelineStage[] { logReg });
		PipelineModel m1 = pL.fit(lblFdf);
		LogisticRegressionTrainingSummary logSum = ((LogisticRegressionModel) (m1.stages()[0])).summary();
		double numMeasure = logSum.accuracy(), numMeasure2 = logSum.weightedFMeasure();
		//print info
		System.out.println();
		System.out.println("Training DataSet Information:");
		//trainingdataset.csv
		System.out.println("Accuracy: " + numMeasure);
		System.out.println("F-measure: " + numMeasure2);

		Dataset<Row> testingDf1 = getDataFrame(s, true, VALIDATION_DATASET).cache(),
				testR = m1.transform(testingDf1);
		//set testing data frame on the result of the getter
		System.out.println("\n Validation Training Information:");
		//validationdataset.csv
		testR.select("features", "label", "prediction").show(5, false);
		calcModel(testR);

		try {
			m1.write().overwrite().save(MODEL_PATH);
		} catch (IOException e) {
			logger.error(e);
		}
	}

 /**
 * Calculates the accuracy, F1, and other metrics for the model.
 * @param predictions the predictions for the model, in the form of a DataFrame containing the
 * predictions for each row.
  */
	public void calcModel(Dataset<Row> predictions) {
		System.out.println();
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
		evaluator.setMetricName("accuracy");
		System.out.println("Model Accuracy: " + evaluator.evaluate(predictions));
		evaluator.setMetricName("f1");
		System.out.println("F1: " + evaluator.evaluate(predictions));
	}

 /**
  * Returns a dataframe of the wine dataset.
  * @param s The spark session to use.
  * @param transform Whether to transform the dataframe to a vector.
  * @param name The name of the file to load.
  * @return The dataframe of the wine dataset.
  */
	public Dataset<Row> getDataFrame(SparkSession s, boolean transform, String name) {

		Dataset<Row> vDF = s.read().format("csv").option("header", "true").option("multiline", true).option("sep", ";")
				.option("quote", "\"").option("dateFormat", "M/d/y").option("inferSchema", true).load(name)
				.withColumnRenamed("fixed acidity", "fixed_acidity")
				.withColumnRenamed("volatile acidity", "volatile_acidity")
				.withColumnRenamed("citric acid", "citric_acid").withColumnRenamed("residual sugar", "residual_sugar")
				.withColumnRenamed("chlorides", "chlorides")
				.withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
				.withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
				.withColumnRenamed("density", "density").withColumnRenamed("pH", "pH")
				.withColumnRenamed("sulphates", "sulphates").withColumnRenamed("alcohol", "alcohol")
				.withColumnRenamed("quality", "label");
		//change to as and clean up

		vDF.show(5);

		Dataset<Row> lblFdf = vDF
				.select("label", "alcohol", "sulphates", "pH", "density", "free_sulfur_dioxide", "total_sulfur_dioxide",
						"chlorides", "residual_sugar", "citric_acid", "volatile_acidity", "fixed_acidity")
				.na().drop().cache();

		if (transform)
			lblFdf = new VectorAssembler().setInputCols(new String[] { "alcohol", "sulphates", "pH", "density",
					"free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar", "citric_acid",
					"volatile_acidity", "fixed_acidity" }).setOutputCol("features").transform(lblFdf)
					.select("label", "features");
		return lblFdf;
	}
}

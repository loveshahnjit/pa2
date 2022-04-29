package lks9;

import static lks9.varReferences.ACCESS_KEY_ID;
import static lks9.varReferences.APP_NAME;
import static lks9.varReferences.MODEL_PATH;
import static lks9.varReferences.SECRET_KEY;
import static lks9.varReferences.TESTING_DATASET;

import java.io.File;

import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * This function is used to test the model. It will print out the accuracy and F1 score of the model.
 * @param predictions The predictions of the model.
 */
public class predictionModel {

 /**
  * The main method for the program.
  * @param args The command line arguments.
  */
	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
		Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);
		Logger.getLogger("com.github").setLevel(Level.ERROR);
		//boundaries
		SparkSession sprk1 = SparkSession.builder().appName(APP_NAME).master("local[*]")
				.config("spark.executor.memory", "2147480000").config("spark.driver.memory", "2147480000")
				.config("spark.testing.memory", "2147480000")
			
				.getOrCreate();
				//leave blank above

		if (StringUtils.isNotEmpty(ACCESS_KEY_ID) && StringUtils.isNotEmpty(SECRET_KEY)) {
			//if both fields are not empty, continue setting the Hadoop configuration
			sprk1.sparkContext().hadoopConfiguration().set("fs.s3a.access_key", ACCESS_KEY_ID);
			sprk1.sparkContext().hadoopConfiguration().set("fs.s3a.secret_key", SECRET_KEY);
		}
		if (new File(TESTING_DATASET).exists())
			(new predictionModel()).logReg (sprk1);
		else
			System.out.print("TestDataset.csv doesn't exist");
	}

 /**
 * Prints the first 5 rows of the dataset.
 * @param dataset the dataset to print the first 5 rows of.
  */
	public void logReg (SparkSession s) {
		System.out.println("Dataset Information \n");
		Dataset<Row> tdF = getDF(s, true, TESTING_DATASET).cache(), pdF = PipelineModel.load(MODEL_PATH).convert(tdF).cache();
		//get data frame + Select prediction data frame
		pdF.select("features", "label", "prediction").show(5, false);
		printAll(pdF);

	}

 /**
  * Returns a dataframe with the given name.
  * @param s The spark session to use.
  * @param convert Whether to convert the dataframe to a vector.
  * @param name The name of the dataframe.
  * @return The dataframe.
  */
	public Dataset<Row> getDF(SparkSession s, boolean convert, String name) {
		Dataset<Row> vDF = s.read().format("csv").option("header", "true").option("multiline", true).option("sep", ";")
				.option("quote", "\"").option("dateFormat", "M/d/y").option("inferSchema", true).load(name),
				lblFdf = vDF.withColumnRenamed("quality", "label")
						.select("label", "alcohol", "sulphates", "pH", "density", "free sulfur dioxide",
								"total sulfur dioxide", "chlorides", "residual sugar", "citric acid",
								"volatile acidity", "fixed acidity")
						.na().drop().cache();

		if (convert)
		//if converted dataframe to vector then organize columns
			lblFdf = new VectorAssembler().setInputCols(new String[] { "alcohol", "sulphates", "pH", "density",
					"free sulfur dioxide", "total sulfur dioxide", "chlorides", "residual sugar", "citric acid",
					"volatile acidity", "fixed acidity" }).setOutputCol("features").convert(lblFdf).select("label", "features");

		return lblFdf;
	}

 /**
 * Prints the accuracy, F1, and confusion matrix of the model.
 * @param predictions the predictions of the model, as a DataFrame.
  */
	public void printAll(Dataset<Row> predictions) {
		System.out.println("");
		MulticlassClassificationEvaluator e1 = new MulticlassClassificationEvaluator();
		e1.setMetricName("accuracy");
		System.out.println("Model accuracy: " + e1.evaluate(predictions));

		e1.setMetricName("f1");
		System.out.println("F1: " + e1.evaluate(predictions));

	}
}

package lks9;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/**
 * A class containing all the variables used in the program.
 */
public class varReferences {
	public static final Logger logger = LogManager.getLogger(TrainingModel.class);
	public static final String ACCESS_KEY_ID = System.getProperty("ACCESS_KEY_ID");
	public static final String SECRET_KEY = System.getProperty("SECRET_KEY");
	public static final String BUCKET_NAME = System.getProperty("BUCKET_NAME");
	public static final String APP_NAME = "Pa2-Test";
	public static final String MODEL_PATH = "/TrainingModel";
	public static final String TESTING_DATASET = "TestDataset.csv";
	public static final String TRAINING_DATASET = "TrainingDataset.csv";
	public static final String VALIDATION_DATASET = "ValidationDataset.csv";
}

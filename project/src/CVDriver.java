import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Random;

/* Performs n-folds cross validation on the classifier
 * Usage: CVDriver [sparse|dense] training_file_name training_label_file_name */
public class CVDriver {

  private static final int NUM_FOLDS = 5;
  private static final Random rand = new Random(484);
  private static final ArrayList<ArrayList<Record>> trainingFolds = new ArrayList<>(NUM_FOLDS);
  private static final ArrayList<ArrayList<Record>> testFolds = new ArrayList<>(NUM_FOLDS);
  private static final Timer timer = new Timer();

  public static void main(String[] args) {
    boolean sparse = args[0].equals("sparse") ? true : false;
    timer.start();
    timer.printElapsedTime("Reading in training records information from " + args[1]);
    ArrayList<Record> trainingRecords = Record.readRecords(args[1], args[2], sparse);
    timer.printElapsedTime("Creating folds");
    createFolds(trainingRecords);
    timer.printElapsedTime("Cross validating");
    crossValidate();
    timer.printElapsedTime("Finished");
  }

  /* Performs cross validation on the created folds */
  private static void crossValidate() {
    ArrayList<Double> accuracies = new ArrayList<>(NUM_FOLDS);
    for(int fold = 0; fold < NUM_FOLDS; fold++) {
      ArrayList<String> predictedLabels = ClassificationDriver.calculateLabels(trainingFolds.get(fold), testFolds.get(fold));
      accuracies.add(calcAccuracy(predictedLabels, testFolds.get(fold)));
      System.out.printf("Fold #%d's Accuracy: %f\n", (fold+1), accuracies.get(fold));
    }
    double mean = DataMiningUtil.mean(accuracies);
    double stdDev = DataMiningUtil.sampleStandardDeviation(accuracies);
    System.out.printf("Accuracy: mean = %f, std.dev = %f\n", mean, stdDev);
    timer.printElapsedTime("Finished.");
  }

  /* Returns the percentage of predictedLabels that match the label for the corresponding
   * testRecord */
  private static double calcAccuracy(ArrayList<String> predictedLabels, ArrayList<Record> testRecords) {
    int misses = 0;
    int hits = 0;
    for(int i = 0; i < predictedLabels.size(); i++) {
      String prediction = predictedLabels.get(i);
      String expected = testRecords.get(i).getClassLabel();
      if(prediction.equals(expected)) {
        hits++;
      } else {
        misses++;
      }
    }
    return (1.0*hits)/(misses+hits);
  }

  /* Splits the training set into NUM_FOLDS number of folds. Uses those
   * splits to create a training set and a testing set for each fold. */
  private static void createFolds(ArrayList<Record> trainingRecords) {
    ArrayList<String> classLabels = new ArrayList<>();
    for(Record record : trainingRecords) {
      classLabels.add(record.getClassLabel());
    }
    ArrayList<ArrayList<Record>> groups = DataMiningUtil.getStratifiedGroups(trainingRecords, NUM_FOLDS, classLabels, rand);
    for(int i = 0; i < groups.size(); i++) {
      ArrayList<Record> trainingFold = new ArrayList<>();
      for(int j = 0; j < groups.size(); j++) {
        if(i != j) {
          trainingFold.addAll(groups.get(j));
        }
      }
      trainingFolds.add(trainingFold);
      testFolds.add(groups.get(i));
    }
  }
}

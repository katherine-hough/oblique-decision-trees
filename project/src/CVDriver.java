import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Random;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.File;

/* Performs n-folds cross validation on the classifier or creates the specified number of folds
 * Usage: CVDriver [sparse|dense] training_file_name training_label_file_name num_folds random_seed [F|GA-ODT|C-DT|DT]]*/
public class CVDriver {

  private static int NUM_FOLDS = 5;
  private static final ArrayList<ArrayList<Record>> trainingFolds = new ArrayList<>(NUM_FOLDS);
  private static final ArrayList<ArrayList<Record>> testFolds = new ArrayList<>(NUM_FOLDS);
  private static final Timer timer = new Timer();

  public static void main(String[] args) {
    timer.start();
    NUM_FOLDS = Integer.parseInt(args[3]);
    boolean sparse = args[0].equals("sparse") ? true : false;
    timer.printElapsedTime("Reading in training records information from " + args[1]);
    ArrayList<Record> trainingRecords = Record.readRecords(args[1], args[2], sparse);
    timer.printElapsedTime(String.format("Creating folds from %d records with %d attributes and %d classes", trainingRecords.size(), Record.getAllFeatures(trainingRecords).size(), Record.getAllClasses(trainingRecords).size()));
    createFolds(trainingRecords, new Random(Integer.parseInt(args[4])));
    if(args[5].equals("F")) {
      timer.printElapsedTime("Writing folds to files");
      writeFoldsToFiles(args[1], trainingRecords, sparse);
    } else {
      timer.printElapsedTime("Cross validating");
      crossValidate(args[5]);
    }
    timer.printElapsedTime("Finished");
  }

  /* Performs cross validation on the created folds */
  private static void crossValidate(String method) {
    ArrayList<Double> accuracies = new ArrayList<>(NUM_FOLDS);
    for(int fold = 0; fold < NUM_FOLDS; fold++) {
      ArrayList<String> predictedLabels = ClassificationDriver.calculateLabels(method, trainingFolds.get(fold), testFolds.get(fold));
      accuracies.add(calcAccuracy(predictedLabels, testFolds.get(fold)));
      System.out.printf("Fold #%d's Accuracy: %.5f\n", (fold+1), accuracies.get(fold));
    }
    double mean = DataMiningUtil.mean(accuracies);
    double stdDev = DataMiningUtil.sampleStandardDeviation(accuracies);
    System.out.printf("Accuracies: %s\n", accuracies);
    System.out.printf("Accuracy: mean = %f, std.dev = %f\n", mean, stdDev);
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
  private static void createFolds(ArrayList<Record> trainingRecords, Random rand) {
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

  private static void writeFoldsToFiles(String trainFilename, ArrayList<Record> trainingRecords, boolean sparse) {
    String[] dataPath = trainFilename.split(Pattern.quote(java.io.File.separator));
    String[] dataFile = (dataPath[dataPath.length-1]).split("\\.");
    TreeSet<Integer> features = new TreeSet<>(Record.getAllFeatures(trainingRecords));
    String directoryPath = "";
    for(int i = 0; i<dataPath.length-1;i++) {
      directoryPath += dataPath[i] + java.io.File.separator;
    }
    directoryPath += "folds" + java.io.File.separator + NUM_FOLDS + "-folds";
    DataMiningUtil.makeDirectoryPath(directoryPath);
    for(int i = 0; i < NUM_FOLDS; i++) {
      String testFile = directoryPath + java.io.File.separator + dataFile[0] + (i+1) + "-test." + dataFile[1];
      String trainFile = directoryPath + java.io.File.separator + dataFile[0] + (i+1) + "-train." + dataFile[1];
      try {
        PrintWriter pw = new PrintWriter(testFile);
        for(Record record : testFolds.get(i)) {
          if(sparse) {
            pw.println(record.toSparseString());
          } else {
            pw.println(record.toDenseString(features));
          }
        }
        pw.close();
        PrintWriter pw2 = new PrintWriter(trainFile);
        for(Record record : trainingFolds.get(i)) {
          if(sparse) {
            pw2.println(record.toSparseString());
          } else {
            pw2.println(record.toDenseString(features));
          }
        }
        pw2.close();
      } catch (IOException e) {
        e.printStackTrace();
        System.err.println("Error occurred writing folds to files.");
      }
    }
  }
}

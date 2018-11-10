import java.util.ArrayList;
import java.util.HashMap;
/* Classifies the test instances using the training instances. Writes the
 * calculated classes out to a file.

 * Usage: ClassificationDriver [sparse|dense] test_file_name training_file_name training_label_file_name output_file_name */
public class ClassificationDriver {

  public static void main(String[] args) {
    Timer timer = new Timer();
    timer.start();
    boolean sparse = args[0].equals("sparse") ? true : false;
    timer.printElapsedTime("Reading in test records information from " + args[1]);
    ArrayList<Record> testRecords = Record.readRecords(args[1], sparse);
    timer.printElapsedTime("Reading in training records information from " + args[2]);
    ArrayList<Record> trainingRecords = Record.readRecords(args[2], args[3], sparse);
    timer.printElapsedTime("Classifying test records.");
    ArrayList<String> calculatedLabels = calculateLabels(trainingRecords, testRecords);
    timer.printElapsedTime("Writing predicted labels to " + args[4]);
    DataMiningUtil.writeToFile(calculatedLabels, args[4]);
    timer.printElapsedTime("Finished.");
  }

  /* Returns a list of the labels calculated for the specified training and test data */
  public static ArrayList<String> calculateLabels(ArrayList<Record> trainingData, ArrayList<Record> testData) {
    Classifier classifier = new ObliqueDecisionTree(trainingData);
    return classifier.classifyAll(testData);
  }
}

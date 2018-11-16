import java.util.ArrayList;
import java.util.HashMap;
/* Classifies the test instances using the training instances. Writes the
 * calculated classes out to a file.

 * Usage: ClassificationDriver [sparse|dense] test_file_name training_file_name training_label_file_name output_file_name [GA-ODT|C-DT|DT]*/
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
    ArrayList<String> calculatedLabels = calculateLabels(args[5], trainingRecords, testRecords);
    timer.printElapsedTime("Writing predicted labels to " + args[4]);
    DataMiningUtil.writeToFile(calculatedLabels, args[4]);
    timer.printElapsedTime("Finished");
  }

  /* Returns a list of the labels calculated for the specified training and test
   * data using the specified decision tree method */
  public static ArrayList<String> calculateLabels(String method, ArrayList<Record> trainingData, ArrayList<Record> testData) {
    DecisionTree classifier;
    if(method.equals("GA-ODT")) {
      classifier = new GeneticDecisionTree(trainingData);
    } else if(method.equals("C-DT")) {
      classifier = new ComplexDecisionTree(trainingData);
    } else if(method.equals("DT")) {
      classifier = new DecisionTree(trainingData);
    } else {
      throw new RuntimeException("Invalid decision tree method: " + method);
    }
    return classifier.classifyAll(testData);
  }
}

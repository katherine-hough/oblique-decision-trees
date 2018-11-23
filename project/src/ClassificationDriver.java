import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

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
    DecisionTreeBuilder builder = new DecisionTreeBuilder()
                      .reservePortionDenom(5)
                      .prune(true)
                      .rand(new Random(484))
                      .numThreads(4)
                      .maxBuckets(100)
                      .maxNonHomogenuousPercent(0.0)
                      .maxBaseConditions(300)
                      .minBaseConditions(100)
                      .baseConditionsPercent(0.01)
                      .maxGeneConditions(50)
                      .minGeneConditions(1)
                      .geneConditionsPercent(0.3)
                      .populationSize(264)
                      .tournamentSize(4)
                      .replacementTournamentSize(7)
                      .maxGenerations(250);
    Class<? extends SplitStrategy> strategyClass;
    if(method.equals("GA-ODT")) {
      strategyClass = GeneticSplitStrategy.class;
    } else if(method.equals("C-DT")) {
      strategyClass = ComplexSplitStrategy.class;
    } else if(method.equals("DT")) {
      strategyClass = SplitStrategy.class;
    } else {
      throw new RuntimeException("Invalid splitting method name: " + method);
    }
    DecisionTree classifier = builder.build(trainingData, strategyClass);
    // System.out.println(TreePrintingUtil.getTreeString(classifier, 5));
    return classifier.classifyAll(testData);
  }
}

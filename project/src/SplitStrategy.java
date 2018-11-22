import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Predicate;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/* Represents some method for splitting a decision tree */
public class SplitStrategy {

  /* Number of threads used in the thread pool */
  protected final int numThreads;
  /* Maximum number of buckets considered for splitting per attribute */
  protected final int maxBuckets;

  /* Default Constructor */
  public SplitStrategy() {
    this.numThreads = 4;
    this.maxBuckets = 100;
  }

  /* 2-arg Constructor */
  public SplitStrategy(int numThreads, int maxBuckets) {
    this.numThreads = numThreads;
    this.maxBuckets = maxBuckets;
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
  public SplitCondition selectSplitCondition(List<Record> records, DecisionTree tree) {
    List<SplitCondition> baseConditions = getBaseConditions(records);
    List<SplitCondition> mostPureConditions = mostPureConditions(1, baseConditions, records, tree);
    if(records.equals(tree.getTrainingRecords())) {
      return mostPureConditions.get(0);
    } else {
      return resolveTiedConditions(mostPureConditions, tree);
    }
  }

  /* Gets the basic set of conditions which split the feature space along the
   * feature axes */
  protected List<SplitCondition> getBaseConditions(List<Record> records) {
    List<Integer> features = new ArrayList<Integer>(Record.getAllFeatures(records));
    List<SplitCondition> conditions = new ArrayList<>();
    for(Integer feature : features) {
      for(double bucket : AttributeSpace.getSplitBuckets(records, feature, maxBuckets)) {
        Predicate<Record> condition = (record) -> {
          return record.getOrDefault(feature) < bucket;
        };
        conditions.add(new SplitCondition(condition, feature, bucket));
      }
    }
    return conditions;
  }

  /* Selects and returns a SplitCondition from the specified list of tied (with respected
   * to their impurities) conditions. */
  protected SplitCondition resolveTiedConditions(List<SplitCondition> ties, DecisionTree tree) {
    if(ties.size() == 0) {
      return null;
    } else if(ties.size() == 1) {
      return ties.get(0);
    } else {
      List<SplitCondition> tiesCopy = new ArrayList<>(ties.size());
      for(SplitCondition tie : ties) {
        tiesCopy.add(tie.copy());
      }
      List<SplitCondition> reEvals = mostPureConditions(1, ties, tree.getTrainingRecords(), tree);
      return reEvals.size() == 0 ? null : reEvals.get(0);
    }
  }

  /* Return a list of the specified number of conditions with the lowest impurity.
   * Includes any additional conditions that are tied for lowest impurity */
  protected List<SplitCondition> mostPureConditions(int numConditions, List<SplitCondition> conditions, List<Record> records, DecisionTree tree) {
    ExecutorService taskExecutor = Executors.newFixedThreadPool(numThreads);
    final int conditionsPerTask = 100;
    ArrayList<Callable<Boolean>> tasks = new ArrayList<>(conditions.size());
    for(int x = 0; x < conditions.size(); x+=conditionsPerTask) {
      final int i = x;
      Callable<Boolean> task = () -> {
        try {
          for(int j = i; j < Math.min(i+conditionsPerTask, conditions.size()); j++) {
            if(conditions.get(j).getImpurity() < 0) {
              conditions.get(j).setImpurity(getTotalGiniImpurity(records, conditions.get(j), tree.getClassIndexMap()));
            }
          }
          return true;
        } catch(Exception e) {
          e.printStackTrace();
          return false;
        }
      };
      tasks.add(task);
    }
    try {
      taskExecutor.invokeAll(tasks);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    } finally {
      taskExecutor.shutdown();
    }
    PriorityQueue<SplitCondition> conditionQueue = new PriorityQueue<>(conditions);
    ArrayList<SplitCondition> topConditions = new ArrayList<>();
    SplitCondition prev = null;
    while(!conditionQueue.isEmpty() && topConditions.size() < numConditions) {
      topConditions.add(conditionQueue.poll());
    }
    return topConditions;
  }

  /* The weighted GINI impurity of the split formed by partitioning the specified
   * list of records based on the specified condition */
  public static double getTotalGiniImpurity(List<Record> records, SplitCondition splitCondition, HashMap<String, Integer> classIndexMap) {
    int[] classFreqsLeft = new int[classIndexMap.size()];
    int[] classFreqsRight = new int[classIndexMap.size()];
    int totalLeft = 0;
    int totalRight = 0;
    for(Record record : records) {
      int index = classIndexMap.get(record.getClassLabel());
      if(splitCondition.test(record)) {
        totalLeft++;
        classFreqsLeft[index]++;
      } else {
        totalRight++;
        classFreqsRight[index]++;
      }
    }
    double giniLeft = getGiniImpurity(classFreqsLeft);
    double giniRight = getGiniImpurity(classFreqsRight);
    double probLeft = (1.0*totalLeft)/records.size();
    double probRight = (1.0*totalRight)/records.size();
    return giniLeft*probLeft + giniRight*probRight;
  }

  /* Calculates the GINI impurity based on the specified array of class frequencies*/
  public static double getGiniImpurity(int[] classFreqs) {
    int sum = 0;
    int total = 0;
    for(int classFreq : classFreqs) {
      sum += classFreq*classFreq;
      total += classFreq;
    }
    return (total == 0) ? 0.0 : (1.0 - sum/(1.0 * total * total));
  }
}

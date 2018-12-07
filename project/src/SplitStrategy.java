import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Predicate;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Collection;

/* Represents some method for splitting a decision tree */
public class SplitStrategy {

  /* Number of threads used in the thread pool */
  private final int numThreads;
  /* Maximum number of buckets considered for splitting per attribute */
  private final int maxBuckets;
  /* Stores that class label at the index it is associated with */
  private HashMap<String, Integer> classIndexMap;

  /* Default Constructor */
  public SplitStrategy(DecisionTreeBuilder builder) {
    this.numThreads = builder.numThreads;
    this.maxBuckets = builder.maxBuckets;
  }

  /* Setter for classIndexMap */
  public void setClassIndexMap(HashMap<String, Integer> classIndexMap) {
    this.classIndexMap = classIndexMap;
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
    ArrayList<Callable<Boolean>> tasks = new ArrayList<>();
    ArrayList<List<SplitCondition>> conditionsList = new ArrayList<>();
    int i = 0;
    for(int feature : features) {
      ArrayList<SplitCondition> next = new ArrayList<>();
      conditionsList.add(next);
      Callable<Boolean> task = () -> {
        try {
          addFeatureBaseConditions(records, feature, next);
          return true;
        } catch(Exception e) {
          e.printStackTrace();
          return false;
        }
      };
      tasks.add(task);
    }
    runTasks(tasks);
    List<SplitCondition> conditions = new ArrayList<>();
    for(List<SplitCondition> list : conditionsList) {
      conditions.addAll(list);
    }
    return conditions;
  }

  /* Gets the basic set of conditions which split the feature space along the
   * the specified feature axis */
  private void addFeatureBaseConditions(List<Record> records, int feature, List<SplitCondition> conditions) {
    int[] classFreqsRight = DecisionTree.getClassFreqs(records, classIndexMap);
    int[] classFreqsLeft = new int[classIndexMap.size()];
    AttributeSpace attrSpace = new AttributeSpace(records, feature, maxBuckets, classIndexMap);
    for(int i = 0; i < attrSpace.numCandidates(); i++) {
      double bucket = attrSpace.getCandidate(i);
      Predicate<Record> condition = (record) -> {
        return record.getOrDefault(feature) < bucket;
      };
      SplitCondition split = new SplitCondition(condition, feature, bucket);
      int[] classFreqs = attrSpace.getCandidatesClassFreqs(i);
      for(int j = 0; j < classFreqs.length; j++) {
        classFreqsLeft[j] += classFreqs[j];
        classFreqsRight[j] -= classFreqs[j];
      }
      split.setImpurity(calcWeightedGiniImpurity(classFreqsLeft, classFreqsRight, DecisionTree.sumArray(classFreqsLeft), records.size()));
      conditions.add(split);
    }
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
    runTasks(tasks);
    PriorityQueue<SplitCondition> conditionQueue = new PriorityQueue<>(conditions);
    ArrayList<SplitCondition> topConditions = new ArrayList<>();
    SplitCondition prev = null;
    while(!conditionQueue.isEmpty() && topConditions.size() < numConditions) {
      topConditions.add(conditionQueue.poll());
    }
    return topConditions;
  }

  /* Runs the specified tasks */
  private <T> void runTasks(Collection<? extends Callable<T>> tasks) {
    ExecutorService taskExecutor = Executors.newFixedThreadPool(numThreads);
    try {
      taskExecutor.invokeAll(tasks);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    } finally {
      taskExecutor.shutdown();
    }
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
    return calcWeightedGiniImpurity(classFreqsLeft, classFreqsRight, totalLeft, records.size());
  }

  /* Returns the weighted GINI impurity of a binary split that results in the specified properties */
  public static double calcWeightedGiniImpurity(int[] classFreqsLeft, int[] classFreqsRight, int totalLeft, int totalRecords) {
    double giniLeft = getGiniImpurity(classFreqsLeft);
    double giniRight = getGiniImpurity(classFreqsRight);
    double probLeft = (1.0*totalLeft)/totalRecords;
    double probRight = (1.0*(totalRecords-totalLeft))/totalRecords;
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

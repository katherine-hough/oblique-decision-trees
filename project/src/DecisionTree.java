import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.Predicate;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/* A trained decision tree used for classifying records. */
public class DecisionTree extends Classifier {

  // Number of threads in the thread pool
  private static final int NUM_THREADS = 4;

  protected static final double DEFAULT_FEATURE_VALUE = 0.0;
  protected static final int MAX_NON_HOMOG = 2;
  protected Classifier leftChild;
  protected Classifier rightChild;
  protected String leafLabel;
  protected List<Record> reachingRecords;
  protected DecisionTree root;
  protected SplitCondition splitCondition;
  protected String defaultClass;


  /* Constructor for the root node calls two argument constructor*/
  public DecisionTree(List<Record> reachingRecords) {
    this(reachingRecords, null);
  }

  /* Classifies a single training instance and returns a string representation of
   * that calculated class */
  public String classify(Record record) {
    if(leafLabel != null) {
      return leafLabel;
    } else if(splitCondition.test(record)) {
      return leftChild.classify(record);
    } else {
      return rightChild.classify(record);
    }
  }

  /* 2-arg Constructor */
  protected DecisionTree(List<Record> reachingRecords, DecisionTree root) {
    this.defaultClass = (root==null) ? getMostFrequentLabel(reachingRecords) : root.defaultClass;
    this.root = root;
    this.reachingRecords = reachingRecords;
    if (reachingRecords.size() == 0) {
      leafLabel = defaultClass;
    } else if(mostlyHomogeneous(reachingRecords)) {
      leafLabel = getMostFrequentLabel(reachingRecords);
    } else {
      splitCondition = selectSplitCondition();
      if(splitCondition == null) {
        leafLabel = getMostFrequentLabel(reachingRecords);
      } else {
        makeChildren();
      }
    }
  }

  /* Create the child nodes for the current node */
  protected void makeChildren() {
    List<Record> trueRecords = new ArrayList<>(reachingRecords);
    List<Record> falseRecords  = splitOnCondition(splitCondition, trueRecords);
    DecisionTree r = (root == null) ? this : root;
    leftChild = new DecisionTree(trueRecords, r);
    rightChild = new DecisionTree(falseRecords, r);
  }

  /* Returns the most common label based on the specified frequency map */
  protected static String getMostFrequentLabel(HashMap<String, Integer> classFreqs) {
    String mostFrequentLabel = null;
    for(String label : classFreqs.keySet()) {
      if(mostFrequentLabel == null || classFreqs.get(label) > classFreqs.get(mostFrequentLabel)) {
        mostFrequentLabel = label;
      }
    }
    return mostFrequentLabel;
  }

  /* Returns the most common label for the records in the specified list */
  protected static String getMostFrequentLabel(List<Record> records) {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(records, (record) -> Collections.singleton(record.getClassLabel()));
    return getMostFrequentLabel(classFreqs);
  }

  /* Returns whether every record in the specified list have the same class label */
  protected static boolean homogeneous(List<Record> records) {
    String first = null;
    for(Record record : records) {
      if(first == null) {
        first = record.getClassLabel();
      } else if (!first.equals(record.getClassLabel())) {
          return false;
      }
    }
    return true;
  }

  /* Returns whether almost every record in the specified list have the same class label */
  protected static boolean mostlyHomogeneous(List<Record> records) {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(records, (record) -> Collections.singleton(record.getClassLabel()));
    String mostFreq = getMostFrequentLabel(classFreqs);
    int minTotal = 0;
    for(String key : classFreqs.keySet()) {
      if(!key.equals(mostFreq)) {
        minTotal += classFreqs.get(key);
      }
    }
    return minTotal <= MAX_NON_HOMOG;
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
  protected SplitCondition selectSplitCondition() {
    ArrayList<Integer> features = new ArrayList<Integer>(Record.getAllFeatures(reachingRecords));
    ArrayList<SplitCondition> conditions = new ArrayList<>(features.size());
    for(Integer feature : features) {
      for(double bucket : Record.getSplitBuckets(reachingRecords, feature, DEFAULT_FEATURE_VALUE)) {
        Predicate<Record> condition = (record) -> {
          return record.getOrDefault(feature, DEFAULT_FEATURE_VALUE) < bucket;
        };
        String desc = String.format("[Feat # %d] < %f", feature, bucket);
        conditions.add(new SplitCondition(desc, condition));
      }
    }
    return mostPureConditions(1, conditions).get(0);
  }

  /* Return a list of the specified number of conditions with the lowest impurity.
   * Includes any additional conditions that are tied for lowest impurity */
  protected ArrayList<SplitCondition> mostPureConditions(int numConditions, ArrayList<SplitCondition> conditions) {
    ExecutorService taskExecutor = Executors.newFixedThreadPool(NUM_THREADS);
    final int conditionsPerTask = 100;
    ArrayList<Callable<Boolean>> tasks = new ArrayList<>(conditions.size());
    for(int x = 0; x < conditions.size(); x+=conditionsPerTask) {
      final int i = x;
      Callable<Boolean> task = () -> {
        try {
          for(int j = i; j < Math.min(i+conditionsPerTask, conditions.size()); j++) {
            conditions.get(j).setImpurity(getTotalGiniImpurity(conditions.get(j)));
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
    double prevImpur = -1;
    int absoluteMax = 3*numConditions;
    while(!conditionQueue.isEmpty()) {
      SplitCondition next = conditionQueue.poll();
      if((numConditions-- <=  -absoluteMax ) || (numConditions <= 0 && (prevImpur!=-1) && (prevImpur < next.getImpurity()))) {
        break;
      } else {
        topConditions.add(next);
        prevImpur = next.getImpurity();
      }
    }
    return topConditions;
  }

  /* The weighted GINI impurity if the records reaching this node are partitioned based on
   * the condition */
  protected double getTotalGiniImpurity(SplitCondition splitCondition) {
    List<Record> containingRecords= new ArrayList<>(reachingRecords);
    List<Record> omittingRecords = splitOnCondition(splitCondition, containingRecords);
    double gini1 = getGiniImpurity(containingRecords);
    double gini2 = getGiniImpurity(omittingRecords);
    double prob1 = (1.0*containingRecords.size())/reachingRecords.size();
    double prob2 = (1.0*omittingRecords.size())/reachingRecords.size();
    return gini1*prob1 + gini2*prob2;
  }

  /* Gets the GINI impurity of the specified list of records */
  protected static double getGiniImpurity(List<Record> records) {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(records, (record) -> Collections.singleton(record.getClassLabel()));
    double sum = 0;
    for(String label : classFreqs.keySet()) {
      sum += (classFreqs.get(label)*classFreqs.get(label));
    }
    return 1.0 - (sum/(records.size()*records.size()));
  }

  /* Removes all records from the specified list that do not contain the specified feature
   * and adds them to the returned list */
  public static ArrayList<Record> splitOnCondition(SplitCondition splitCondition, List<Record> records) {
    Iterator<Record> it = records.iterator();
    ArrayList<Record> falseRecords = new ArrayList<>();
    while(it.hasNext()) {
      Record record = it.next();
      if(!splitCondition.test(record)) {
        it.remove();
        falseRecords.add(record);
      }
    }
    return falseRecords;
  }

  /* Accessor for the node's leftChild */
  public Classifier getLeftChild() {
    return leftChild;
  }

  /* Accessor for the node's rightChild */
  public Classifier getRightChild() {
    return rightChild;
  }

  /* Accessor for the node's splitCondition */
  public SplitCondition getSplitCondition() {
    return splitCondition;
  }

  /* Accessor for the node's leftLabel */
  public String getLeafLabel() {
    return leafLabel;
  }
}

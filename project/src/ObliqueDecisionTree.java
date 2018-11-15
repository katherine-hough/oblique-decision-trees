import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.Random;

/* A DecisionTree that allows splits to be made on oblqiue axes */
public class ObliqueDecisionTree extends DecisionTree {

  // Maximum number of most pure base conditions considered
  private static final int MAX_BASE_CONDITIONS = 50;
  // Size of population in the genetic algorithm
  private static final int POP_SIZE = 264;
  // Maximum number of generations run in the genetic algorithm
  private static final int MAX_GENS = 200;
  // Used to generate random numbers
  private static final Random RAND = new Random(848);
  // Maximum number of buckets considered for splitting per attribute
  private static final int MAX_BUCKETS = 100;

  /* Constructor for the root node calls two argument constructor*/
  public ObliqueDecisionTree(List<Record> reachingRecords) {
    super(reachingRecords);
  }

  /* 2-arg Constructor */
  protected ObliqueDecisionTree(List<Record> reachingRecords, DecisionTree root) {
    super(reachingRecords, root);
  }

  /* Create the child nodes for the current node */
  @Override
  protected void makeChildren() {
    System.out.println(splitCondition);
    List<Record> trueRecords = new ArrayList<>(reachingRecords);
    List<Record> falseRecords  = splitOnCondition(splitCondition, trueRecords);
    DecisionTree r = (root == null) ? this : root;
    leftChild = new ObliqueDecisionTree(trueRecords, r);
    rightChild = new ObliqueDecisionTree(falseRecords, r);
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  protected SplitCondition selectSplitCondition() {
    GeneticAlgorithmSplitter GASplitter = new GeneticAlgorithmSplitter(reachingRecords, targetFeatures(), POP_SIZE, RAND, MAX_BUCKETS);
    return GASplitter.getBestSplitCondition(MAX_GENS);
  }

  /* Returns an array of features to be considered by the genetic algorithm.
   * These feature are the features that would have resulted in the purest traditional
   * decision tree split. */
  private int[] targetFeatures() {
    int maxBaseConditions = Math.min(reachingRecords.size()/10+1, MAX_BASE_CONDITIONS);
    TreeSet<Integer> features = new TreeSet<>(Record.getAllFeatures(reachingRecords));
    if(features.size() > maxBaseConditions) {
      HashMap<SplitCondition, Pair<Integer, Double>> baseConditions = getBaseConditions();
      ArrayList<SplitCondition> mostPureConditions = mostPureConditions(maxBaseConditions, new ArrayList<>(baseConditions.keySet()));
      baseConditions.keySet().retainAll(mostPureConditions);
      features = new TreeSet<>();
      for(Pair<Integer, Double> featureBucketPair : baseConditions.values()) {
        features.add(featureBucketPair.getKey());
      }
    }
    int[] targetFeatures = new int[features.size()];
    for(int i = 0; i < targetFeatures.length; i++) {
      targetFeatures[i] = features.pollFirst();
    }
    return targetFeatures;
  }

  /* Gets the basic set of initial conditions which split the feature space
   * along the feature axes */
  private HashMap<SplitCondition, Pair<Integer, Double>> getBaseConditions() {
    ArrayList<Integer> features = new ArrayList<Integer>(Record.getAllFeatures(reachingRecords));
    HashMap<SplitCondition, Pair<Integer, Double>> conditions = new HashMap<>();
    for(Integer feature : features) {
      for(double bucket : AttributeSpace.getSplitBuckets(reachingRecords, feature, MAX_BUCKETS)) {
        Predicate<Record> condition = (record) -> {
          return record.getOrDefault(feature) < bucket;
        };
        String desc = String.format("[#%d]<%2.2f", feature, bucket);
        conditions.put(new SplitCondition(desc, condition), new Pair<Integer, Double>(feature, bucket));
      }
    }
    return conditions;
  }
}

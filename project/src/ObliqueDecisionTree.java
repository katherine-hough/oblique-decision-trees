import java.util.List;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.Iterator;
import java.util.HashSet;
import java.util.function.Predicate;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

/* A DecisionTree that allows splits to be made on oblqiue axes */
public class ObliqueDecisionTree extends DecisionTree {

  private static final double MIN_SPARSE_FEATURE_PERCENT = 0.05;

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
    ArrayList<SplitCondition> conditions = getBaseConditions();
    int maxCond = Math.min(300, (int)(conditions.size()*.001)+100);
    conditions = mostPureConditions(maxCond, conditions);
    conditions.addAll(getSecondaryConditions(conditions));
    conditions = mostPureConditions(maxCond, conditions);
    conditions.addAll(getSecondaryConditions(conditions));
    conditions = mostPureConditions(1, conditions);
    return conditions.isEmpty() ? null : conditions.get(0);
  }

  /*Creates conditions which are combinations of the specified conditions */
  protected ArrayList<SplitCondition> getSecondaryConditions(ArrayList<SplitCondition> conditions) {
    ArrayList<SplitCondition> secondaryConditions = new ArrayList<>();
    for(int i = 0; i < conditions.size(); i++) {
      for(int j = i+1; j < conditions.size(); j++) {
        SplitCondition condition1 = conditions.get(i);
        SplitCondition condition2 = conditions.get(j);
        SplitCondition or = condition1.or(condition2);
        SplitCondition and = condition1.and(condition2);
        SplitCondition notOr = (condition1.negate()).and(condition2);
        SplitCondition notAnd = (condition1.negate()).and(condition2);
        secondaryConditions.add(or);
        secondaryConditions.add(and);
        secondaryConditions.add(notOr);
        secondaryConditions.add(notAnd);
      }
    }
    return secondaryConditions;
  }

  /* Gets the basic set of initial conditions which split the feature space
   * along the feature axes */
  private ArrayList<SplitCondition> getBaseConditions() {
    ArrayList<Integer> features = new ArrayList<Integer>(Record.getAllFeatures(reachingRecords));
    HashMap<Integer, Integer> featureFreqs = DataMiningUtil.createFreqMap(reachingRecords, (record) -> record.keySet());
    features.removeIf((feature) -> featureFreqs.get(feature) < MIN_SPARSE_FEATURE_PERCENT*reachingRecords.size());
    ArrayList<SplitCondition> conditions = new ArrayList<>(features.size());
    for(Integer feature : features) {
      for(double bucket : Record.getSplitBuckets(reachingRecords, feature)) {
        Predicate<Record> condition = (record) -> {
          return record.getOrDefault(feature, Record.DEFAULT_FEATURE_VALUE) < bucket;
        };
        String desc = String.format("[#%d]<%2.2f", feature, bucket);
        conditions.add(new SplitCondition(desc, condition));
      }
    }
    return conditions;
  }
}

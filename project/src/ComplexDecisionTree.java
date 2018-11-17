import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;

/* A DecisionTree that allows splits to be made that consider multiple features */
public class ComplexDecisionTree extends DecisionTree {

  // Maximum number of buckets considered for splitting per attribute
  private static final int MAX_BUCKETS = 100;

  /* Constructor for the root node calls two argument constructor*/
  public ComplexDecisionTree(List<Record> trainingRecords) {
    super(trainingRecords);
  }

  /* 2-arg Constructor */
  protected ComplexDecisionTree(List<Record> reachingRecords, DecisionTree root) {
    super(reachingRecords, root);
  }

  /* Creates a child node of the same class */
  @Override
  protected ComplexDecisionTree makeChild(List<Record> records, DecisionTree root) {
    return new ComplexDecisionTree(records, root);
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
    return resolveTiedConditions(conditions);
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
}

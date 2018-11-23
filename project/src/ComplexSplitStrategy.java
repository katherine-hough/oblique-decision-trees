import java.util.List;
import java.util.ArrayList;

/* Splits decision trees on conditions that consider boolean combinations of multiple
 * features */
public class ComplexSplitStrategy extends SplitStrategy {

  /* Maximum number of conditions used to create secondary conditions */
  private final int maxBaseConditions;
  /* Minimum number of conditions used to create secondary conditions */
  private final int minBaseConditions;
  /* Percentage of total records added to the minimum number of conditions */
  private final double baseConditionsPercent;

  /* Default Constructor */
  public ComplexSplitStrategy(DecisionTreeBuilder builder) {
    super(builder);
    this.maxBaseConditions = builder.maxBaseConditions;
    this.minBaseConditions = builder.minBaseConditions;
    this.baseConditionsPercent = builder.baseConditionsPercent;
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  public SplitCondition selectSplitCondition(List<Record> records, DecisionTree tree) {
    List<SplitCondition> conditions = getBaseConditions(records);
    int numCond = Math.min(maxBaseConditions, (int)(conditions.size()*baseConditionsPercent)+minBaseConditions);
    conditions = mostPureConditions(numCond, conditions, records, tree);
    conditions.addAll(getSecondaryConditions(conditions));
    conditions = mostPureConditions(numCond, conditions, records, tree);
    conditions.addAll(getSecondaryConditions(conditions));
    conditions = mostPureConditions(1, conditions, records, tree);
    return resolveTiedConditions(conditions, tree);
  }

  /* Creates conditions which are combinations of the specified conditions */
  private List<SplitCondition> getSecondaryConditions(List<SplitCondition> conditions) {
    List<SplitCondition> secondaryConditions = new ArrayList<>();
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

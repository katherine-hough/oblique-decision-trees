import java.util.List;
import java.util.TreeSet;

/* Splits decision trees on conditions that consider boolean combinations of multiple
 * features */
public class GeneticSplitStrategy extends SplitStrategy {

  /* Maximum number of conditions considered in the genetic algorithm */
  private final int maxGeneConditions;
  /* Minimum number of conditions considered in the genetic algorithm */
  private final int minGeneConditions;
  /* Percentage of total records added to the minimum number of conditions in the
   * genetic algorithm */
  private final double geneConditionsPercent;
  /* Used to build genetic algorithm splitter. */
  private GeneticSplitter.GeneticSplitterBuilder geneticBuilder;

  /* Default Constructor */
  public GeneticSplitStrategy(DecisionTreeBuilder builder) {
    super(builder);
    this.maxGeneConditions = builder.maxGeneConditions;
    this.minGeneConditions = builder.minGeneConditions;
    this.geneConditionsPercent = builder.geneConditionsPercent;
    this.geneticBuilder = builder.geneticBuilder;
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  public SplitCondition selectSplitCondition(List<Record> records, DecisionTree tree) {
    setTargetFeatures(records, tree);
    GeneticSplitter GASplitter = geneticBuilder.records(records)
                                        .classIndexMap(tree.getClassIndexMap())
                                        .build();
    return GASplitter.getBestSplitCondition();
  }

  /* Sets the targets features and top conditions used by the splitter. These features
   * and conditions are the ones that would have resulted in the purest traditional
   * decision tree split. */
  private void setTargetFeatures(List<Record> records, DecisionTree tree) {
    int numCond = (int)Math.min(records.size()*geneConditionsPercent + minGeneConditions, maxGeneConditions);
    List<SplitCondition> conditions = mostPureConditions(numCond, getBaseConditions(records), records, tree);
    TreeSet<Integer> features = new TreeSet<>();
    for(SplitCondition condition : conditions) {
      features.add(condition.getFeature());
    }
    int[] targetFeatures = new int[features.size()];
    for(int i = 0; i < targetFeatures.length; i++) {
      targetFeatures[i] = features.pollFirst();
    }
    geneticBuilder = geneticBuilder.targetFeatures(targetFeatures).topConditions(conditions);
  }
}

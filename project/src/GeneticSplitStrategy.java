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
  private final GeneticSplitter.GeneticSplitterBuilder geneticBuilder;

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
    GeneticSplitter GASplitter = geneticBuilder.records(records)
                                        .targetFeatures(targetFeatures(records, tree))
                                        .classIndexMap(tree.getClassIndexMap())
                                        .build();
    return GASplitter.getBestSplitCondition();
  }

  /* Returns an array of features to be considered by the genetic algorithm.
   * These feature are the features that would have resulted in the purest traditional
   * decision tree split. */
  private int[] targetFeatures(List<Record> records, DecisionTree tree) {
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
    return targetFeatures;
  }
}

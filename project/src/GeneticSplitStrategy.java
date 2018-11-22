import java.util.List;
import java.util.TreeSet;
import java.util.Random;

/* Splits decision trees on conditions that consider boolean combinations of multiple
 * features */
public class GeneticSplitStrategy extends SplitStrategy {

  /* Maximum number of conditions considered in the genetic algorithm */
  private final int maxBaseConditions;
  /* Minimum number of conditions considered in the genetic algorithm */
  private final int minBaseConditions;
  /* Percentage of total records added to the minimum number of conditions */
  private final double baseConditionsPercent;
  /* Used to build genetic algorithm splitter. Created by the root node and shared by
   * all nodes in the same tree */
  private final GeneticSplitter.GeneticSplitterBuilder builder;

  /* Default Constructor */
  public GeneticSplitStrategy() {
    super();
    this.maxBaseConditions = 50;
    this.minBaseConditions = 1;
    this.baseConditionsPercent = 0.3;
    this.builder = new GeneticSplitter.GeneticSplitterBuilder()
                      .rand(new Random(848))
                      .populationSize(264)
                      .tournamentSize(4)
                      .replacementTournamentSize(7)
                      .maxBuckets(maxBuckets)
                      .maxGenerations(250);
  }

  /* 2-arg Constructor */
  public GeneticSplitStrategy(int numThreads, int maxBuckets, int maxBaseConditions, int minBaseConditions, double baseConditionsPercent, GeneticSplitter.GeneticSplitterBuilder builder) {
    super(numThreads, maxBuckets);
    this.maxBaseConditions = maxBaseConditions;
    this.minBaseConditions = minBaseConditions;
    this.baseConditionsPercent = baseConditionsPercent;
    this.builder = builder.maxBuckets(maxBuckets);
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  public SplitCondition selectSplitCondition(List<Record> records, DecisionTree tree) {
    GeneticSplitter GASplitter = builder.records(records)
                                        .targetFeatures(targetFeatures(records, tree))
                                        .classIndexMap(tree.getClassIndexMap())
                                        .build();
    return GASplitter.getBestSplitCondition();
  }

  /* Returns an array of features to be considered by the genetic algorithm.
   * These feature are the features that would have resulted in the purest traditional
   * decision tree split. */
  private int[] targetFeatures(List<Record> records, DecisionTree tree) {
    int numCond = (int)Math.min(records.size()*baseConditionsPercent + minBaseConditions, maxBaseConditions);
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

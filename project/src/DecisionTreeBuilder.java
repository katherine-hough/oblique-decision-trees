import java.util.Random;
import java.util.List;

/* Builder for creating DecisionTree instances */
public class DecisionTreeBuilder {

  /* THe portion of nodes reserved when the tree is pruned */
  public int reservePortionDenom;
  /* Whether or not this tree should be pruned */
  public boolean prune;
  /* Used for the generation of any random element */
  public Random rand;
  /* Number of threads used in the thread pool */
  public int numThreads;
  /* Maximum number of buckets considered for splitting per attribute */
  public int maxBuckets;
  /* Maximum percent of records reaching the node that can be from a different
   * class for the node to still be considered homogeneous.*/
  public double maxNonHomogenuousPercent;
  /* Maximum number of conditions used to create secondary conditions. Used by
   * CompoundSplitStrategy. */
  public int maxBaseConditions;
  /* Minimum number of conditions used to create secondary conditions. Used by
   * CompoundSplitStrategy. */
  public int minBaseConditions;
  /* Percentage of total records added to the minimum number of conditions. Used by
   * CompoundSplitStrategy. */
  public double baseConditionsPercent;
  /* Maximum number of conditions considered in the genetic algorithm. Used by
   * GeneticSplitStrategy*/
  public int maxGeneConditions;
  /* Minimum number of conditions considered in the genetic algorithm. Used by
   * GeneticSplitStrategy*/
  public int minGeneConditions;
  /* Percentage of total records added to the minimum number of conditions in the
   * genetic algorithm. Used by GeneticSplitStrategy*/
  public double geneConditionsPercent;
  /* Used to build genetic algorithm splitter by GeneticSplitStrategy */
  public GeneticSplitter.GeneticSplitterBuilder geneticBuilder;

  /* Constructor, sets all values to their defaults */
  public DecisionTreeBuilder() {
    this.reservePortionDenom = 5;
    this.prune = false;
    this.rand = new Random(484);
    this.numThreads = 4;
    this.maxBuckets = 200;
    this.maxNonHomogenuousPercent = 0.001;
    this.maxBaseConditions = 300;
    this.minBaseConditions = 100;
    this.baseConditionsPercent = 0.01;
    this.maxGeneConditions = 100;
    this.minGeneConditions = 1;
    this.geneConditionsPercent = 0.45;
    this.geneticBuilder = new GeneticSplitter.GeneticSplitterBuilder()
                      .rand(rand)
                      .populationSize(128)
                      .tournamentSize(4)
                      .replacementTournamentSize(6)
                      .maxBuckets(maxBuckets)
                      .maxGenerations(200);
  }

  public DecisionTreeBuilder reservePortionDenom(int reservePortionDenom) {
    this.reservePortionDenom = reservePortionDenom;
    return this;
  }

  public DecisionTreeBuilder prune(boolean prune) {
    this.prune = prune;
    return this;
  }

  public DecisionTreeBuilder rand(Random rand) {
    this.rand = rand;
    this.geneticBuilder.rand(rand);
    return this;
  }

  public DecisionTreeBuilder numThreads(int numThreads) {
    this.numThreads = numThreads;
    return this;
  }

  public DecisionTreeBuilder maxBuckets(int maxBuckets) {
    this.maxBuckets = maxBuckets;
    this.geneticBuilder.maxBuckets(maxBuckets);
    return this;
  }

  public DecisionTreeBuilder maxNonHomogenuousPercent(double maxNonHomogenuousPercent) {
    this.maxNonHomogenuousPercent = maxNonHomogenuousPercent;
    return this;
  }

  public DecisionTreeBuilder maxBaseConditions(int maxBaseConditions) {
    this.maxBaseConditions = maxBaseConditions;
    return this;
  }

  public DecisionTreeBuilder minBaseConditions(int minBaseConditions) {
    this.minBaseConditions = minBaseConditions;
    return this;
  }

  public DecisionTreeBuilder baseConditionsPercent(double baseConditionsPercent) {
    this.baseConditionsPercent = baseConditionsPercent;
    return this;
  }

  public DecisionTreeBuilder maxGeneConditions(int maxGeneConditions) {
    this.maxGeneConditions = maxGeneConditions;
    return this;
  }

  public DecisionTreeBuilder minGeneConditions(int minGeneConditions) {
    this.minGeneConditions = minGeneConditions;
    return this;
  }

  public DecisionTreeBuilder geneConditionsPercent(double geneConditionsPercent) {
    this.geneConditionsPercent = geneConditionsPercent;
    return this;
  }

  public DecisionTreeBuilder populationSize(int populationSize) {
    this.geneticBuilder.populationSize(populationSize);
    return this;
  }

  public DecisionTreeBuilder tournamentSize(int tournamentSize) {
    this.geneticBuilder.tournamentSize(tournamentSize);
    return this;
  }

  public DecisionTreeBuilder replacementTournamentSize(int replacementTournamentSize) {
    this.geneticBuilder.replacementTournamentSize(replacementTournamentSize);
    return this;
  }

  public DecisionTreeBuilder maxGenerations(int maxGenerations) {
    this.geneticBuilder.maxGenerations(maxGenerations);
    return this;
  }

  /* Returns a DecisionTree instance built from the builder's parameters */
  public <T extends SplitStrategy> DecisionTree build(List<Record> records, Class<T> strategyClass) {
    try {
      SplitStrategy splitStrategy = strategyClass.getConstructor(DecisionTreeBuilder.class).newInstance(this);
      DecisionTree tree = new DecisionTree(records, (int)(records.size()*maxNonHomogenuousPercent)+1, splitStrategy);
      if(prune) {
        tree.pruneTree(reservePortionDenom, rand);
      }
      return tree;
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException("Failed to build tree's strategy.");
    }
  }
}

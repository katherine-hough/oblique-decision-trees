import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.Random;

/* A DecisionTree that allows splits to be made on oblique axes */
public class GeneticDecisionTree extends DecisionTree {

  /* Maximum number of conditions considered in the genetic algorithm */
  private final int maxBaseConditions = 50;
  /* Minimum number of conditions considered in the genetic algorithm */
  private final int minBaseConditions = 1;
  /* Percentage of total records added to the minimum number of conditions */
  private final double baseConditionsPercent = 0.3;

  /* Used to build genetic algorithm splitter. Created by the root node and shared by
   * all nodes in the same tree */
  private GeneticSplitter.GeneticSplitterBuilder builder;

  /* Constructor for the root node calls two argument constructor*/
  public GeneticDecisionTree(List<Record> trainingRecords) {
    super(trainingRecords);
  }

  /* 2-arg Constructor called by all nodes */
  public GeneticDecisionTree(List<Record> reachingRecords, DecisionTree root) {
    super(reachingRecords, root);
  }

  /* Sets the fields that are created by the root and shared between all nodes
   * or are passed into the constructor*/
  @Override
  protected void setBasicFields(List<Record> reachingRecords, DecisionTree root) {
    super.setBasicFields(reachingRecords, root);
    if(root == null || !(root instanceof GeneticDecisionTree)) {
      this.builder = new GeneticSplitter.GeneticSplitterBuilder()
                      .classIndexMap(classIndexMap)
                      .rand(new Random(848))
                      .populationSize(264)
                      .maxBuckets(MAX_BUCKETS)
                      .tournamentSize(4)
                      .replacementTournamentSize(7)
                      .maxGenerations(250);
    } else {
      this.builder = ((GeneticDecisionTree)root).builder;
    }
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  protected SplitCondition selectSplitCondition() {
    GeneticSplitter GASplitter = builder.records(reachingRecords)
                                        .targetFeatures(targetFeatures())
                                        .build();
    return GASplitter.getBestSplitCondition();
  }

  /* Returns an array of features to be considered by the genetic algorithm.
   * These feature are the features that would have resulted in the purest traditional
   * decision tree split. */
  private int[] targetFeatures() {
    int numCond = (int)Math.min(reachingRecords.size()*baseConditionsPercent + minBaseConditions, maxBaseConditions);
    ArrayList<SplitCondition> conditions = mostPureConditions(numCond, getBaseConditions());
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

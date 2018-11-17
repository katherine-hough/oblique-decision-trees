import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.Random;

/* A DecisionTree that allows splits to be made on oblique axes */
public class GeneticDecisionTree extends DecisionTree {

  /* Maximum number of most pure base conditions considered */
  private static final int MAX_BASE_CONDITIONS = 50;
  /* Maximum number of buckets considered for splitting per attribute */
  private static final int MAX_BUCKETS = 100;
  /* Used to build genetic algorithm splitter. Created by the root node and shared by
   * all nodes in the same tree */
  private GeneticSplitter.GeneticSplitterBuilder builder;

  /* Constructor for the root node calls two argument constructor*/
  public GeneticDecisionTree(List<Record> trainingRecords) {
    super(trainingRecords);
  }

  /* 2-arg Constructor */
  protected GeneticDecisionTree(List<Record> reachingRecords, DecisionTree root) {
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

  /* Creates a child node of the same class */
  @Override
  protected GeneticDecisionTree makeChild(List<Record> records, DecisionTree root) {
    return new GeneticDecisionTree(records, root);
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
    int maxBaseConditions = Math.min(reachingRecords.size()/3 + 1, MAX_BASE_CONDITIONS);
    ArrayList<SplitCondition> conditions = mostPureConditions(maxBaseConditions, getBaseConditions());
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

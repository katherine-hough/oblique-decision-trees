import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.HashSet;
import java.lang.reflect.InvocationTargetException;

/* Creates a decision tree that reserves a portion of the training instances
 * before creating the tree to use in pruning. Determines an alpha value from the
 * this partial tree and uses it to prune a tree created on the full training set */
public class PrunedTreeCreator {

  /* Randomly (based on the specified Random instance) reserves 1 divided by the
   * specfied reserve portion denominator of the training data to use in
   * post-pruning. Creates a tree of the specified tree class with the remaining
   * portion of the training data. Prunes that tree and return it. */
  public static <T extends DecisionTree> T createTree(Class<T> treeClass, List<Record> trainingRecords, int reservePortionDenom, Random rand)
  throws InstantiationException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
    T fullTree = treeClass.getConstructor(List.class).newInstance(new ArrayList<>(trainingRecords));
    List<Record> reservedRecords = selectReservedRecords(trainingRecords, reservePortionDenom, rand);
    trainingRecords.removeAll(reservedRecords);
    T alphaSelectTree = treeClass.getConstructor(List.class).newInstance(trainingRecords);
    double alpha = selectAlpha(alphaSelectTree, reservedRecords);
    pruneTree(fullTree, alpha);
    return fullTree;
  }

  /* Prunes leaves from the decision tree select the split based on the specified alpha value */
  private static void pruneTree(DecisionTree decisionTree, double selectedAlpha) {
    ArrayList<DecisionTree> pruneNodes = new ArrayList<>();
    ArrayList<Double> alphas = new ArrayList<>();
    while(decisionTree.getLeafLabel() == null) {
      double minAlpha = -1;
      DecisionTree pruneNode = null;
      for(DecisionTree node : decisionTree.getAllNonLeaves()) {
        double alpha = (node.nodeTrainingError()-node.subtreeTrainingError())/(node.getAllLeaves().size() - 1);
        if(pruneNode == null || minAlpha > alpha) {
          minAlpha = alpha;
          pruneNode = node;
        }
      }
      alphas.add(minAlpha);
      pruneNodes.add(pruneNode);
      pruneNode.prune();
    }
    int bestIndex = -1;
    for(int i = 0; i < alphas.size(); i++) {
      if(alphas.get(i) > selectedAlpha) {
        break;
      }
      bestIndex = i;
    }
    for(int i = bestIndex+1; i < pruneNodes.size(); i++) {
      pruneNodes.get(i).unprune();
    }
    if(bestIndex != -1) {
      System.out.println("Selected prune index: " + bestIndex);
    }
  }

  /* Prunes leaves from the decision tree. Returns the alpha value of the best prune. */
  private static double selectAlpha(DecisionTree decisionTree, List<Record> reservedRecords) {
    int unprunedCorrectPredictions = calculateCorrectPredictions(decisionTree, reservedRecords);
    ArrayList<DecisionTree> pruneNodes = new ArrayList<>();
    ArrayList<Integer> correctPredictions = new ArrayList<>();
    ArrayList<Double> alphas = new ArrayList<>();
    while(decisionTree.getLeafLabel() == null) {
      double minAlpha = -1;
      DecisionTree pruneNode = null;
      for(DecisionTree node : decisionTree.getAllNonLeaves()) {
        double alpha = (node.nodeTrainingError()-node.subtreeTrainingError())/(node.getAllLeaves().size() - 1);
        if(pruneNode == null || minAlpha > alpha) {
          minAlpha = alpha;
          pruneNode = node;
        }
      }
      alphas.add(minAlpha);
      pruneNodes.add(pruneNode);
      correctPredictions.add(calculateCorrectPredictions(decisionTree, reservedRecords));
      pruneNode.prune();
    }
    int bestIndex = -1;
    for(int i = 0; i < correctPredictions.size(); i++) {
      if(bestIndex == -1 || correctPredictions.get(i) > correctPredictions.get(bestIndex)) {
        bestIndex = i;
      }
    }
    if(unprunedCorrectPredictions >= correctPredictions.get(bestIndex)) {
      bestIndex = -1;
    }
    for(int i = bestIndex+1; i < correctPredictions.size(); i++) {
      pruneNodes.get(i).unprune();
    }
    return bestIndex == -1 ? -1 : alphas.get(bestIndex);
  }

  /* Returns the number of records that the specified decision tree correctly classifies
   * from the specified records */
  private static int calculateCorrectPredictions(DecisionTree decisionTree, List<Record> reservedRecords) {
    int correctPredictions = 0;
    for(Record reservedRecord : reservedRecords) {
      String prediction = decisionTree.classify(reservedRecord);
      if(prediction.equals(reservedRecord.getClassLabel())) {
        correctPredictions++;
      }
    }
    return correctPredictions;
  }

  /* Returns a portion of specified records to reserve */
  private static List<Record> selectReservedRecords(List<Record> records, int reservePortionDenom, Random rand) {
    ArrayList<String> classLabels = new ArrayList<>();
    for(Record record : records) {
      classLabels.add(record.getClassLabel());
    }
    ArrayList<ArrayList<Record>> groups = DataMiningUtil.getStratifiedGroups(records, reservePortionDenom, classLabels, rand);
    return groups.get(0);
  }
}

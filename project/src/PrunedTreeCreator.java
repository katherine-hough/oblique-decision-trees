import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.HashSet;
import java.lang.reflect.InvocationTargetException;

/* Creates a decision tree that reserves a portion of the training instances
 * before creating the tree to use in pruning. */
public class PrunedTreeCreator {

  /* Randomly (based on the specified Random instance) reserves 1 divided by the
   * specfied reserve portion denominator of the training data to use in
   * post-pruning. Creates a tree of the specified tree class with the remaining
   * portion of the training data. Prunes that tree and return it. */
  public static <T extends DecisionTree> T createTree(Class<T> treeClass, List<Record> trainingRecords, int reservePortionDenom, Random rand)
  throws InstantiationException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
    List<Record> reservedRecords = selectReservedRecords(trainingRecords, reservePortionDenom, rand);
    trainingRecords.removeAll(reservedRecords);
    System.out.println("Reserved Records: " + reservedRecords.size());
    T decisionTree = treeClass.getConstructor(List.class).newInstance(trainingRecords);
    pruneTree(decisionTree, reservedRecords);
    return decisionTree;
  }

  /* Prunes leaves from the decision tree. */
  private static void pruneTree(DecisionTree decisionTree, List<Record> reservedRecords) {
    double bestPredict = calculateCorrectPredictions(decisionTree, reservedRecords);
    if(bestPredict == reservedRecords.size()) return;
    DecisionTree bestPruneCandidate = null;
    HashSet<DecisionTree> candidateNodes =  getCandidateNodes(decisionTree);
    for(DecisionTree node : candidateNodes) {
      node.prune();
      double predict = calculateCorrectPredictions(decisionTree, reservedRecords);
      if(predict > bestPredict) {
        bestPredict = predict;
        bestPruneCandidate = node;
      }
      node.unprune();
    }
    if(bestPruneCandidate != null) {
      System.out.println("Pruned: " + bestPruneCandidate);
      bestPruneCandidate.prune();
    }
  }

  /* Returns every non-root node that is a parent of at least one leaf nodes. */
  private static HashSet<DecisionTree> getCandidateNodes(DecisionTree decisionTree) {
    HashSet<DecisionTree> candidateNodes = new HashSet<>();
    ArrayList<DecisionTree> nodesToVisit = new ArrayList<>();
    nodesToVisit.add(decisionTree);
    while(!nodesToVisit.isEmpty()) {
      DecisionTree cur= nodesToVisit.remove(0);
      if(cur.getLeftChild() != null) {
        if(cur.getLeftChild().getLeftChild() == null && cur.getLeftChild().getRightChild() == null) {
          if(cur != decisionTree) {
            candidateNodes.add(cur);
          }
        } else {
          nodesToVisit.add(cur.getLeftChild());
        }
      }
      if(cur.getRightChild() != null) {
        if(cur.getRightChild().getLeftChild() == null && cur.getRightChild().getRightChild() == null) {
          if(cur != decisionTree) {
            candidateNodes.add(cur);
          }
        } else {
          nodesToVisit.add(cur.getRightChild());
        }
      }
    }
    return candidateNodes;
  }

  /* Returns the number of records that the specified decision tree correct classifies
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

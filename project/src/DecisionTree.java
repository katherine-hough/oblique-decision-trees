import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.function.Predicate;
import java.util.Random;

/* A trained decision tree used for classifying records. */
public class DecisionTree extends Classifier {

  /* Maps each different class label to a different integer index. */
  private final HashMap<String, Integer> classIndexMap;
  /* Stores that class label at the index it is associated with */
  private final String[] indexClassMap;
  /* Class returned if a node is empty. */
  private final String defaultClass;
  /* Maximum number of records reaching the node that can be from a different
   * class for the node to still be considered homogeneous.*/
  private final int maxNonHomogenuousRecords;
  /* The root node of the decision tree */
  private final DecisionNode root;
  /* Used to split node */
  private final SplitStrategy splitStrategy;
  /* The records used to train this tree */
  private final List<Record> trainingRecords;
  /* References this tree so that inner class can pass it to the splitStrategy */
  private final DecisionTree tree;

  /* Constructor */
  public DecisionTree(List<Record> records, int maxNonHomogenuousRecords, SplitStrategy splitStrategy) {
    this.tree = this;
    this.trainingRecords = records;
    this.classIndexMap = new HashMap<>();
    this.splitStrategy = splitStrategy;
    for(Record record : records) {
      this.classIndexMap.putIfAbsent(record.getClassLabel(), classIndexMap.size());
    }
    this.indexClassMap = new String[classIndexMap.size()];
    for(String key : classIndexMap.keySet()) {
      this.indexClassMap[classIndexMap.get(key)] = key;
    }
    this.defaultClass = getMostFrequentLabel(records);
    this.maxNonHomogenuousRecords = maxNonHomogenuousRecords;
    this.root = new DecisionNode(records);
  }

  /* Classifies a single training instance and returns a string representation of
   * that calculated class */
  public String classify(Record record) {
    return root.classify(record);
  }

  /* Returns the frequencies of the different classes found in the specified list
   * of records */
  private int[] getClassFreqs(List<Record> records) {
    int[] classFreqs = new int[classIndexMap.size()];
    for(Record record : records) {
      classFreqs[classIndexMap.get(record.getClassLabel())]++;
    }
    return classFreqs;
  }

  /* Returns the index of the largest value in the specified array */
  private int getIndexOfMax(int[] values) {
    int maxIndex = 0;
    for(int i = 1; i < values.length; i++) {
      maxIndex = (values[i] > values[maxIndex]) ? i : maxIndex;
    }
    return maxIndex;
  }

  /* Returns the sum of the values in the specified array */
  private int getSum(int[] values) {
    int sum = 0;
    for(int i = 0; i < values.length; i++) {
      sum += values[i];
    }
    return sum;
  }

  /* Returns the class label with the highest frequency in the specified list of
   * records */
  private String getMostFrequentLabel(List<Record> records) {
    int[] classFreqs = getClassFreqs(records);
    return indexClassMap[getIndexOfMax(classFreqs)];
  }

  /* Returns the number of records not from the majority class in the specified list */
  private int getNumberMisclassified(List<Record> records) {
    int[] classFreqs = getClassFreqs(records);
    return getSum(classFreqs) - classFreqs[getIndexOfMax(classFreqs)];
  }

  /* Removes all records from the specified list that do not contain the specified feature
   * and adds them to the returned list */
  private static ArrayList<Record> splitOnCondition(SplitCondition splitCondition, List<Record> records) {
    Iterator<Record> it = records.iterator();
    ArrayList<Record> falseRecords = new ArrayList<>();
    while(it.hasNext()) {
      Record record = it.next();
      if(!splitCondition.test(record)) {
        it.remove();
        falseRecords.add(record);
      }
    }
    return falseRecords;
  }

  /* Accessor for classIndexMap */
  public HashMap<String, Integer> getClassIndexMap() {
    return classIndexMap;
  }

  /* Accessor for trainingRecords */
  public List<Record> getTrainingRecords() {
    return trainingRecords;
  }

  /* Creates lists containing the string representations of nodes at each level
   of the tree */
  public ArrayList<ArrayList<ArrayList<String>>> getBFSStrings() {
    ArrayList<ArrayList<ArrayList<String>>> levels = new ArrayList<>();
    ArrayList<DecisionNode> curLevel = new ArrayList<>();
    ArrayList<String> nullStrings = new ArrayList<>();
    nullStrings.add(null);
    nullStrings.add(null);
    curLevel.add(root);
    while(!curLevel.isEmpty()) {
      ArrayList<DecisionNode> nextLevel = new ArrayList<>();
      ArrayList<ArrayList<String>> curLevelStrings = new ArrayList<>();
      boolean nonNullChild = false;
      for(DecisionNode node : curLevel) {
        if(node == null) {
          nextLevel.add(null);
          nextLevel.add(null);
          curLevelStrings.add(nullStrings);
        } else {
          nextLevel.add(node.leftChild);
          nextLevel.add(node.rightChild);
          curLevelStrings.add(node.getStrings());
          if(node.leftChild != null || node.rightChild != null) {
            nonNullChild = true;
          }
        }
      }
      levels.add(curLevelStrings);
      curLevel = nextLevel;
      if(!nonNullChild) {
        return levels;
      }
    }
    return levels;
  }

  /* Randomly (based on the specified Random instance) reserves 1 divided by the
   * specfied reserve portion denominator instances of the training data to use in
   * post-pruning. Creates a tree of the specified tree class with the remaining
   * portion of the training data. Prunes this tree based on the alpha value from
   * that tree */
  public void pruneTree(int reservePortionDenom, Random rand) {
    List<Record> reservedRecords = selectReservedRecords(trainingRecords, reservePortionDenom, rand);
    List<Record> remainingRecords = new ArrayList<>(trainingRecords);
    remainingRecords.removeAll(reservedRecords);
    DecisionTree alphaSelectTree = new DecisionTree(remainingRecords, maxNonHomogenuousRecords, splitStrategy);
    double alpha = selectAlpha(alphaSelectTree, reservedRecords);
    pruneTree(alpha);
  }

  /* Prunes leaves from this decision tree based on the specified alpha value */
  private void pruneTree(double selectedAlpha) {
    ArrayList<DecisionNode> pruneNodes = new ArrayList<>();
    ArrayList<Double> alphas = new ArrayList<>();
    while(root.leafLabel == null) {
      double minAlpha = -1;
      DecisionNode pruneNode = null;
      for(DecisionNode node : root.getAllNonLeaves()) {
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
    DecisionNode root = decisionTree.root;
    int unprunedCorrectPredictions = calculateCorrectPredictions(decisionTree, reservedRecords);
    ArrayList<DecisionNode> pruneNodes = new ArrayList<>();
    ArrayList<Integer> correctPredictions = new ArrayList<>();
    ArrayList<Double> alphas = new ArrayList<>();
    while(root.leafLabel == null) {
      double minAlpha = -1;
      DecisionNode pruneNode = null;
      for(DecisionNode node : root.getAllNonLeaves()) {
        double alpha = (node.nodeTrainingError()-node.subtreeTrainingError())/(node.getAllLeaves().size() - 1);
        if(pruneNode == null || minAlpha > alpha) {
          minAlpha = alpha;
          pruneNode = node;
        }
      }
      alphas.add(minAlpha);
      pruneNodes.add(pruneNode);
      pruneNode.prune();
      correctPredictions.add(calculateCorrectPredictions(decisionTree, reservedRecords));
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

  /* Returns a portion of specified records to reserve */
  private static List<Record> selectReservedRecords(List<Record> records, int reservePortionDenom, Random rand) {
    ArrayList<String> classLabels = new ArrayList<>();
    for(Record record : records) {
      classLabels.add(record.getClassLabel());
    }
    ArrayList<ArrayList<Record>> groups = DataMiningUtil.getStratifiedGroups(records, reservePortionDenom, classLabels, rand);
    return groups.get(0);
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

  /* Represents a node in the decision tree */
  private class DecisionNode {

    /* This node's leftchild. leftChild is null for leaf nodes */
    DecisionNode leftChild;
    /* This node's right child. rightChild is null for leaf nodes */
    DecisionNode rightChild;
    /* The label associated with this node. leafLabel is null for non-leaf nodes */
    String leafLabel;
    /* Condition used to split the records at this node */
    SplitCondition splitCondition;
    /* The frequencies of the classes of records reaching this node */
    int[] classFreqs;

    /* Constructor */
    DecisionNode(List<Record> reachingRecords) {
      this.classFreqs = getClassFreqs(reachingRecords);
      if (reachingRecords.size() == 0) {
        leafLabel = defaultClass;
      } else if(getNumberMisclassified(reachingRecords) <= maxNonHomogenuousRecords) {
        leafLabel = getMostFrequentLabel(reachingRecords);
      } else {
        splitCondition = splitStrategy.selectSplitCondition(reachingRecords, tree);
        if(splitCondition == null) {
          leafLabel = getMostFrequentLabel(reachingRecords);
        } else {
          List<Record> trueRecords = new ArrayList<>(reachingRecords);
          List<Record> falseRecords  = splitOnCondition(splitCondition, trueRecords);
          leftChild = new DecisionNode(trueRecords);
          rightChild = new DecisionNode(falseRecords);
        }
      }
    }

    /* Classifies a single training instance and returns a string representation of
     * that calculated class */
    String classify(Record record) {
      if(leafLabel != null) {
        return leafLabel;
      } else if(splitCondition.test(record)) {
        return leftChild.classify(record);
      } else {
        return rightChild.classify(record);
      }
    }

    /* Returns String representations of the node */
    ArrayList<String> getStrings() {
      HashMap<String, Integer> classFreqsMap = new HashMap<>();
      for(int i = 0; i < classFreqs.length; i++) {
        classFreqsMap.put(indexClassMap[i], classFreqs[i]);
      }
      ArrayList<String> strs = new ArrayList<>();
      strs.add(classFreqsMap.toString());
      if(splitCondition == null) {
        strs.add("");
      } else {
        strs.add(splitCondition.toString());
      }
      return strs;
    }

    /* Makes this node into a leaf by setting its leafLabel*/
    void prune() {
      leafLabel = indexClassMap[getIndexOfMax(classFreqs)];
    }

    /* If this node has children, restore its status as a non-leaf by setting its
     * leafLabel to null. Throws an exception if the node lacks children */
    void unprune() {
      if(rightChild == null || leftChild == null) {
        throw new RuntimeException("Cannot unprune childless node.");
      }
      leafLabel = null;
    }

    /* Returns the sum of the training errors of all leaf descendants of this node */
    double subtreeTrainingError() {
      double sum = 0;
      for(DecisionNode leaf : getAllLeaves()) {
        sum += leaf.nodeTrainingError();
      }
      return sum;
    }

    /* Returns the training error for this node without considering its children */
    double nodeTrainingError() {
      return SplitStrategy.getGiniImpurity(classFreqs);
    }

    /* Returns every descendant of this node that is a leaf node plus this node if
     * this node is a leaf */
    ArrayList<DecisionNode> getAllLeaves() {
      HashSet<DecisionNode> leaves = new HashSet<>();
      ArrayList<DecisionNode> nodesToVisit = new ArrayList<>();
      nodesToVisit.add(this);
      while(!nodesToVisit.isEmpty()) {
        DecisionNode cur = nodesToVisit.remove(0);
        if(cur.leafLabel == null) {
          nodesToVisit.add(cur.leftChild);
          nodesToVisit.add(cur.rightChild);
        } else {
          leaves.add(cur);
        }
      }
      return new ArrayList<DecisionNode>(leaves);
    }

    /* Returns every descendant of this node that is not a leaf node plus this node if
     * this node is not a leaf */
    ArrayList<DecisionNode> getAllNonLeaves() {
      HashSet<DecisionNode> nonLeaves = new HashSet<>();
      ArrayList<DecisionNode> nodesToVisit = new ArrayList<>();
      nodesToVisit.add(this);
      while(!nodesToVisit.isEmpty()) {
        DecisionNode cur = nodesToVisit.remove(0);
        if(cur.leafLabel == null) {
          nonLeaves.add(cur);
          nodesToVisit.add(cur.leftChild);
          nodesToVisit.add(cur.rightChild);
        }
      }
      return new ArrayList<DecisionNode>(nonLeaves);
    }
  }
}

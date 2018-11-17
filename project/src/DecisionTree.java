import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.Predicate;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/* A trained decision tree used for classifying records. */
public class DecisionTree extends Classifier {

  /* Number of threads used in the thread pool */
  private static final int NUM_THREADS = 4;
  /* Maximum number of buckets considered for splitting per attribute */
  private static final int MAX_BUCKETS = 100;
  /* Maximum percent of the total records that can be from a different class
   * for the node to still be considered homogeneous */
  protected static final double MAX_NON_HOMOG_PERCENT = 0.02;
  /* This node's leftchild. leftChild is null for leaf nodes */
  protected DecisionTree leftChild;
  /* This node's right child. rightChild is null for leaf nodes */
  protected DecisionTree rightChild;
  /* The label associated with this node. leafLabel is null for non-leaf nodes */
  protected String leafLabel;
  /* The training records which reach this node */
  protected List<Record> reachingRecords;
  /* The root node of the decision tree that this node is in. root is null for the
   * root node itself */
  protected DecisionTree root;
  /* Condition used to split the records at this node */
  protected SplitCondition splitCondition;
  /* Number of attributes for the entire dataset. Created by the root node and shared by
   * all nodes in the same tree */
  protected int numAttributes;
  /* Class returned if a node is empty. Created by the root node and shared by
   * all nodes in the same tree */
  protected String defaultClass;
  /* Maps each different class label to a different integer index. Created by the root
   * node and shared by all nodes in the same tree */
  protected HashMap<String, Integer> classIndexMap;
  /* Maximum number of records reaching the node that can be from a different
   * class for the node to still be considered homogeneous. Created by the root
   * node and shared by all nodes in the same tree */
  protected int maxNonHomogenuousRecords;

  /* Constructor for the root node calls two argument constructor*/
  public DecisionTree(List<Record> trainingRecords) {
    this(trainingRecords, null);
  }

  /* 2-arg Constructor called by all nodes */
  protected DecisionTree(List<Record> reachingRecords, DecisionTree root) {
    setBasicFields(reachingRecords, root);
    if (reachingRecords.size() == 0) {
      leafLabel = defaultClass;
    } else if(homogeneous(reachingRecords)) {
      leafLabel = getMostFrequentLabel(reachingRecords);
    } else {
      splitCondition = selectSplitCondition();
      if(splitCondition == null) {
        leafLabel = getMostFrequentLabel(reachingRecords);
      } else {
        // System.out.println(this);
        // System.out.println(splitCondition);
        // Make child nodes
        List<Record> trueRecords = new ArrayList<>(reachingRecords);
        List<Record> falseRecords  = splitOnCondition(splitCondition, trueRecords);
        DecisionTree r = (root == null) ? this : root;
        leftChild = makeChild(trueRecords, r);
        rightChild = makeChild(falseRecords, r);
      }
    }
  }

  /* Sets the fields that are created by the root and shared between all nodes
   * or are passed into the constructor*/
  protected void setBasicFields(List<Record> reachingRecords, DecisionTree root) {
    this.defaultClass = (root==null) ? getMostFrequentLabel(reachingRecords) : root.defaultClass;
    this.classIndexMap = (root==null) ? createClassIndexMap(reachingRecords) : root.classIndexMap;
    this.root = root;
    this.reachingRecords = reachingRecords;
    this.numAttributes = (root==null) ? Record.getAllFeatures(reachingRecords).size() : root.numAttributes;
    this.maxNonHomogenuousRecords = (root==null) ? (int)(reachingRecords.size()*MAX_NON_HOMOG_PERCENT)+1 : root.maxNonHomogenuousRecords;
  }

  /* Classifies a single training instance and returns a string representation of
   * that calculated class */
  public String classify(Record record) {
    if(leafLabel != null) {
      return leafLabel;
    } else if(splitCondition.test(record)) {
      return leftChild.classify(record);
    } else {
      return rightChild.classify(record);
    }
  }

  /* Creates a maps from each different class label to a different integer index */
  public static HashMap<String, Integer> createClassIndexMap(List<Record> records) {
    HashMap<String, Integer> classIndexMap = new HashMap<>();
    for(Record record : records) {
      classIndexMap.putIfAbsent(record.getClassLabel(), classIndexMap.size());
    }
    return classIndexMap;
  }

  /* Creates a child node of the same class */
  protected DecisionTree makeChild(List<Record> records, DecisionTree root) {
    return new DecisionTree(records, root);
  }

  /* Returns the most common label based on the specified frequency map */
  public static String getMostFrequentLabel(HashMap<String, Integer> classFreqs) {
    String mostFrequentLabel = null;
    for(String label : classFreqs.keySet()) {
      if(mostFrequentLabel == null || classFreqs.get(label) > classFreqs.get(mostFrequentLabel)) {
        mostFrequentLabel = label;
      }
    }
    return mostFrequentLabel;
  }

  /* Returns the most common label for the records in the specified list */
  public static String getMostFrequentLabel(List<Record> records) {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(records, (record) -> Collections.singleton(record.getClassLabel()));
    return getMostFrequentLabel(classFreqs);
  }

  /* Returns whether almost every record in the specified list have the same class label */
  protected boolean homogeneous(List<Record> records) {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(records, (record) -> Collections.singleton(record.getClassLabel()));
    String mostFreq = getMostFrequentLabel(classFreqs);
    int minTotal = 0;
    for(String key : classFreqs.keySet()) {
      if(!key.equals(mostFreq)) {
        minTotal += classFreqs.get(key);
      }
    }
    return minTotal <= maxNonHomogenuousRecords;
  }

  /* Gets the basic set of conditions which split the feature space along the
   * feature axes */
  protected ArrayList<SplitCondition> getBaseConditions() {
    ArrayList<Integer> features = new ArrayList<Integer>(Record.getAllFeatures(reachingRecords));
    ArrayList<SplitCondition> conditions = new ArrayList<>();
    for(Integer feature : features) {
      for(double bucket : AttributeSpace.getSplitBuckets(reachingRecords, feature, MAX_BUCKETS)) {
        Predicate<Record> condition = (record) -> {
          return record.getOrDefault(feature) < bucket;
        };
        conditions.add(new SplitCondition(condition, feature, bucket));
      }
    }
    return conditions;
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
  protected SplitCondition selectSplitCondition() {
    return resolveTiedConditions(mostPureConditions(1, getBaseConditions()));
  }

  /* Selected and returns a SplitCondition from the specified list of tied (with respected
   * to their impurities) conditions. */
  protected SplitCondition resolveTiedConditions(ArrayList<SplitCondition> ties) {
    if(ties.size() == 0) {
      return null;
    } else if(ties.size() == 1 || this.root==null) {
      return ties.get(0);
    } else {
      ArrayList<SplitCondition> tiesCopy = new ArrayList<>(ties.size());
      for(SplitCondition tie : ties) {
        tiesCopy.add(tie.copy());
      }
      ArrayList<SplitCondition> reEvals = this.root.mostPureConditions(1, ties);
      return reEvals.size() == 0 ? null : reEvals.get(0);
    }
  }

  /* Return a list of the specified number of conditions with the lowest impurity.
   * Includes any additional conditions that are tied for lowest impurity */
  protected ArrayList<SplitCondition> mostPureConditions(int numConditions, ArrayList<SplitCondition> conditions) {
    ExecutorService taskExecutor = Executors.newFixedThreadPool(NUM_THREADS);
    final int conditionsPerTask = 100;
    ArrayList<Callable<Boolean>> tasks = new ArrayList<>(conditions.size());
    for(int x = 0; x < conditions.size(); x+=conditionsPerTask) {
      final int i = x;
      Callable<Boolean> task = () -> {
        try {
          for(int j = i; j < Math.min(i+conditionsPerTask, conditions.size()); j++) {
            if(conditions.get(j).getImpurity() < 0) {
              conditions.get(j).setImpurity(getTotalGiniImpurity(reachingRecords, conditions.get(j), classIndexMap));
            }
          }
          return true;
        } catch(Exception e) {
          e.printStackTrace();
          return false;
        }
      };
      tasks.add(task);
    }
    try {
      taskExecutor.invokeAll(tasks);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    } finally {
      taskExecutor.shutdown();
    }
    PriorityQueue<SplitCondition> conditionQueue = new PriorityQueue<>(conditions);
    ArrayList<SplitCondition> topConditions = new ArrayList<>();
    SplitCondition prev = null;
    while(!conditionQueue.isEmpty() && topConditions.size() < numConditions) {
      topConditions.add(conditionQueue.poll());
    }
    return topConditions;
  }

  /* The weighted GINI impurity of the split formed by partitioning the specified
   * list fo record based on the specified condition */
  public static double getTotalGiniImpurity(List<Record> records, SplitCondition splitCondition, HashMap<String, Integer> classIndexMap) {
    int[] classFreqsLeft = new int[classIndexMap.size()];
    int[] classFreqsRight = new int[classIndexMap.size()];
    int totalLeft = 0;
    int totalRight = 0;
    for(Record record : records) {
      int index = classIndexMap.get(record.getClassLabel());
      if(splitCondition.test(record)) {
        totalLeft++;
        classFreqsLeft[index]++;
      } else {
        totalRight++;
        classFreqsRight[index]++;
      }
    }
    double giniLeft = getGiniImpurity(classFreqsLeft);
    double giniRight = getGiniImpurity(classFreqsRight);
    double probLeft = (1.0*totalLeft)/records.size();
    double probRight = (1.0*totalRight)/records.size();
    return giniLeft*probLeft + giniRight*probRight;
  }

  /* Calculates the GINI impurity based on the specified array of class frequencies*/
  public static double getGiniImpurity(int[] classFreqs) {
    int sum = 0;
    int total = 0;
    for(int classFreq : classFreqs) {
      sum += classFreq*classFreq;
      total += classFreq;
    }
    return (total == 0) ? 0.0 : (1.0 - sum/(1.0 * total * total));
  }

  /* Removes all records from the specified list that do not contain the specified feature
   * and adds them to the returned list */
  public static ArrayList<Record> splitOnCondition(SplitCondition splitCondition, List<Record> records) {
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

  /* Returns a nicely formatted representation of this node */
  @Override
  public String toString() {
    HashMap<String, Integer> classFreqs = DataMiningUtil.createFreqMap(reachingRecords, (record) -> Collections.singleton(record.getClassLabel()));
    return String.format("%s", classFreqs);
  }

  /* Accessor for the node's leftChild */
  public DecisionTree getLeftChild() {
    return leftChild;
  }

  /* Accessor for the node's rightChild */
  public DecisionTree getRightChild() {
    return rightChild;
  }

  /* Accessor for the node's splitCondition */
  public SplitCondition getSplitCondition() {
    return splitCondition;
  }

  /* Accessor for the node's leftLabel */
  public String getLeafLabel() {
    return leafLabel;
  }

  /* Print the decision tree */
  public void printTree() {
    int maxLevels = 5;
    ArrayList<ArrayList<DecisionTree>> levels = breadthFirstTraversal(maxLevels);
    int[] longest = getLongestStringPerLevel(levels);
    int numLines = 3*(levels.get(levels.size()-1).size()*2-1);
    String[] lines = new String[numLines];
    for(int i = 0; i < lines.length; i++) {
      lines[i] = "";
    }
    String[][][] levelLines = getLevelLines(levels, longest);
    for(int l = levels.size()-1; l>=0; l--) {
      int target = (int)Math.pow(2, levels.size()-1-l);
      int offset = 2*target;
      target--;
      int n = 0;
      int bandHeight = target;
      int bandTarget = (int)Math.pow(2, levels.size()-l);
      int bandOffset = 2*bandTarget;
      bandTarget--;
      for(int i = 0; i < lines.length; i+=3) {
        if((i/3) == target) {
          lines[i] += levelLines[l][n][0];
          lines[i+1] += levelLines[l][n][1];
          lines[i+2] += levelLines[l][n++][2];
          target += offset;
        } else {
          String end = " ";
          String end2 = " ";
          if(l != 0) {
            if(((i/3)>= bandTarget-bandHeight) && ((i/3)<= bandTarget+bandHeight)) {
              if(n > 0 && n < levels.get(l).size() && levels.get(l).get(n-1) != null && levels.get(l).get(n) != null) {
                end = "|";
                end2 = "|";
                if(i/3 == bandTarget) {
                  end2 = "+";
                }
              }
              if(i/3 == bandTarget+bandHeight) {
                bandTarget += bandOffset;
              }
            }
          }
          lines[i] += centerStringAtWidth("", longest[l], " ") + end;
          lines[i+1] += centerStringAtWidth("", longest[l], " ") + end2;
          lines[i+2] += centerStringAtWidth("", longest[l], " ") + end;
        }
      }
    }
    for(String line : lines) {
      System.out.println(line);
    }
  }

  /* Returns the lines for each node on each level */
  private String[][][] getLevelLines(ArrayList<ArrayList<DecisionTree>> levels, int[] longest) {
    String[][][] levelLines = new String[levels.size()][][];
    for(int l = 0; l< levels.size(); l++) {
      ArrayList<DecisionTree> level = levels.get(l);
      levelLines[l] = new String[level.size()][3];
      for(int n = 0; n < level.size(); n++) {
        DecisionTree node = level.get(n);
        String cond = "";
        String freqs = "";
        String divide = " ";
        if(node!=null) {
          divide = "-";
          freqs = node.toString();
          if(node.splitCondition != null) {
            cond = node.splitCondition.toString();
          }
        }
        String end1 = (l==0 || node == null) ? " " : "|";
        String end2 = (l==0 || node == null)? " " : "+";
        levelLines[l][n][0] = centerStringAtWidth(freqs, longest[l], " ") + (n%2==0 ? " " : end1);
        levelLines[l][n][1] = centerStringAtWidth("", longest[l], divide) + end2;
        levelLines[l][n][2] = centerStringAtWidth(cond, longest[l], " ") + (n%2==0 ? end1 : " ");
      }
    }
    return levelLines;
  }

  /* Returns the longest string length at each level of the tree. */
  private int[] getLongestStringPerLevel(ArrayList<ArrayList<DecisionTree>> levels) {
    int[] longest = new int[levels.size()];
    for(int i = 0; i < levels.size(); i++) {
      for(DecisionTree tree : levels.get(i)) {
        if(tree != null) {
          longest[i] = Math.max(longest[i], tree.toString().length());
          if(tree.splitCondition != null) {
            longest[i] = Math.max(longest[i], tree.splitCondition.toString().length());
          }
        }
      }
    }
    return longest;
  }

  /* Returns the specfied string but padded up to the specified minimum width with
   * the specified symbol and centered in that padding */
  private String centerStringAtWidth(String s, int width, String symbol) {
    if(s.length() >= width) {
      return s; // no padding necessary
    }
    int padding = width - s.length();
    int frontPadding = padding/2;
    int backPadding = (padding+1)/2;
    String ret = "";
    for(int i = 0; i < frontPadding; i++) {
      ret += symbol;
    }
    ret +=s;
    for(int i = 0; i < backPadding; i++) {
      ret += symbol;
    }
    return ret;
  }

  /* Creates lists containing the nodes at each level of the tree for up to the
   * specified maximum number of levels */
  private ArrayList<ArrayList<DecisionTree>> breadthFirstTraversal(int maxLevels) {
    ArrayList<ArrayList<DecisionTree>> levels = new ArrayList<>();
    levels.add(new ArrayList<DecisionTree>());
    levels.get(0).add(this);
    int i = 0;
    while(levels.size() > i) {
      ArrayList<DecisionTree> nextLevel = new ArrayList<>();
      boolean nonNullChild = false;
      for(DecisionTree tree : levels.get(i)) {
        if(tree != null) {
          nextLevel.add(tree.leftChild);
          nextLevel.add(tree.rightChild);
          if(tree.leftChild != null || tree.rightChild != null) {
            nonNullChild = true;
          }
        } else {
          nextLevel.add(null);
          nextLevel.add(null);
        }
      }
      if(nonNullChild) {
        levels.add(nextLevel);
      }
      i++;
      if(i > maxLevels) {
        System.out.printf("------> Showing only first %d levels of the tree <------\n", maxLevels);
        return levels;
      }
    }
    return levels;
  }
}

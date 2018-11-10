import java.util.List;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.Iterator;
import java.util.HashSet;
import java.util.Arrays;

/* A DecisionTree that allows splits to be made on oblqiue axes */
public class ObliqueDecisionTree extends DecisionTree {

  /* Constructor for the root node calls two argument constructor*/
  public ObliqueDecisionTree(List<Record> reachingRecords) {
    super(reachingRecords);
  }

  /* 2-arg Constructor */
  protected ObliqueDecisionTree(List<Record> reachingRecords, DecisionTree root) {
    super(reachingRecords, root);
  }

  /* Create the child nodes for the current node */
  @Override
  protected void makeChildren() {
    List<Record> trueRecords = new ArrayList<>(reachingRecords);
    List<Record> falseRecords  = splitOnCondition(splitCondition, trueRecords);
    DecisionTree r = (root == null) ? this : root;
    System.out.printf("%s\nleft: %d|right: %d\n", splitCondition, trueRecords.size(), falseRecords.size());
    leftChild = new ObliqueDecisionTree(trueRecords, r);
    rightChild = new ObliqueDecisionTree(falseRecords, r);
  }

  /* Returns the split condition that produces the purest partition of the reaching
   * records */
   @Override
  protected SplitCondition selectSplitCondition() {
    return super.selectSplitCondition();
  }
}

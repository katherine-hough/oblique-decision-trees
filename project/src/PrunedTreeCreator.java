import java.util.List;
import java.util.ArrayList;
import java.lang.reflect.InvocationTargetException;
import java.util.Random;

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
    return treeClass.getConstructor(List.class).newInstance(trainingRecords);
  }

  private static 

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

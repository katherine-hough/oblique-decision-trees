import java.util.HashMap;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.PriorityQueue;
import java.util.Collection;

public class AttributeSpace {

  private HashMap<String, Integer> classIndexMap;
  private double[] candidates;
  private int[][] freqLists;

  /* Constructor */
  public AttributeSpace(Collection<Record> records, int attribute, int maxBuckets, HashMap<String, Integer> classIndexMap) {
    this.classIndexMap = classIndexMap;
    Random rand = new Random(attribute*records.size());
    TreeMap<Double, FreqList> valueMap = new TreeMap<>();
    for(Record record : records) {
      double value = record.getOrDefault(attribute);
      valueMap.putIfAbsent(value, new FreqList());
      valueMap.get(value).addRecord(record);
    }
    ArrayList<Double> keys = new ArrayList<>(valueMap.keySet());
    int[] shuffledIndexes = shuffledIndexes(valueMap.size()-1, rand);
    PriorityQueue<Integer> selectedIndexes = new PriorityQueue<>();
    int i = 0;
    while(selectedIndexes.size() < maxBuckets && i < shuffledIndexes.length) {
      int index = shuffledIndexes[i++];
      int homog1 = valueMap.get(keys.get(index)).homogenuous();
      int homog2 = valueMap.get(keys.get(index+1)).homogenuous();
      if(!(homog1!=-1 && homog2 != -1 && (homog1 == -2 || homog2 == -2 || homog1 == homog2))) {
        selectedIndexes.add(index);
      }
    }
    candidates = new double[selectedIndexes.size()];
    freqLists = new int[selectedIndexes.size()][classIndexMap.size()];
    int[] freqs = new int[classIndexMap.size()];
    int index = 0;
    int l = 0;
    for(int j = 0; j < keys.size(); j++) {
      double curVal = keys.get(j);
      freqs = addArrays(freqs, valueMap.get(curVal).freqs);
      if(!selectedIndexes.isEmpty() && j == selectedIndexes.peek()) {
        selectedIndexes.poll();
        double nextVal = keys.get(j+1);
        candidates[index] = 0.5*(curVal+nextVal);
        freqLists[index++] = freqs;
      }
    }
  }

  /* Returns an array that is the sum of the two specified arrays */
  public static int[] addArrays(int[] arr1, int[] arr2) {
    int[] sum = new int[arr1.length];
    for(int i = 0; i < arr1.length; i++) {
      sum[i] = arr1[i] + arr2[i];
    }
    return sum;
  }

  /* Returns an array of the values in [0, numIndexes) in a random ordering. Uses
   * Fisher–Yates shuffle */
  public static int[] shuffledIndexes(int numIndexes, Random rand) {
    int[] result = new int[numIndexes];
    for(int i = 0; i < result.length; i++) {
      result[i] = i;
    }
    for(int i = result.length-1; i > 0; i--) {
      int j = rand.nextInt(i+1);
      int temp = result[i];
      result[i] = result[j];
      result[j] = temp;
    }
    return result;
  }

  /* Returns the ith candidate double */
  public double getCandidate(int index) {
    return candidates[index];
  }

  /* Returns the total number of candidates */
  public int numCandidates() {
    return candidates.length;
  }

  /* Returns the class frequency list of all instance with a value of the target
   * attribute less than the ith candidate */
  public int[] getFreqList(int index) {
    return freqLists[index];
  }

  private class FreqList {
    int[] freqs;

    FreqList() {
      this.freqs = new int[classIndexMap.size()];
    }

    void addRecord(Record record) {
      freqs[classIndexMap.get(record.getClassLabel())]++;
    }

    /* Returns -1 is more than one class is present, return -2 if no class is present
     * otherwise returns the index of the only class present */
    int homogenuous() {
      int found = -2;
      for(int i = 0; i < freqs.length; i++) {
        if(freqs[i] != 0) {
          if(found != -2) {
            return -1;
          } else {
            found = i;
          }
        }
      }
      return found;
    }
  }
}

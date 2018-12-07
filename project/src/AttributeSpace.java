import java.util.PriorityQueue;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Objects;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.List;

public class AttributeSpace {

  /* Buckets representing the range of values for a particular attribute */
  ArrayList<ValueBucket> buckets;
  /* Maps each different class label to a different integer index. */
  HashMap<String, Integer> classIndexMap;

  /* Constructor. Makes a sorted list of valueBuckets for splitting the specified records
   * on the specified feature */
  public AttributeSpace(Iterable<Record> records, int attribute, int maxBuckets, Random rand, HashMap<String, Integer> classIndexMap) {
    this.classIndexMap = classIndexMap;
    this.buckets = createInitialBuckets(records, attribute);
    mergeHomogenuousBuckets(buckets);
    reduceBuckets(buckets, maxBuckets, rand);
  }

  /* Returns a sorted list of buckets representing every possible value of the specified attribute */
  private ArrayList<ValueBucket> createInitialBuckets(Iterable<Record> records, int attribute) {
    HashMap<Double, ValueBucket> intialBuckets = new HashMap<>();
    for(Record record : records) {
      double value = record.getOrDefault(attribute);
      intialBuckets.putIfAbsent(value, new ValueBucket(value, value));
      ValueBucket bucket = intialBuckets.get(value);
      int prevFreq = bucket.getOrDefault(record.getClassLabel(), 0);
      bucket.put(record.getClassLabel(), prevFreq+1);
    }
    ArrayList<ValueBucket> values = new ArrayList<>(intialBuckets.values());
    Collections.sort(values);
    return values;
  }

  /* Merges together buckets which are homogenous with respect to the same class */
  private void mergeHomogenuousBuckets(ArrayList<ValueBucket> buckets) {
    Iterator<ValueBucket> iterator = buckets.iterator();
    ValueBucket prev = null;
    while(iterator.hasNext()) {
      ValueBucket cur = iterator.next();
      if(prev != null) {
        if(prev.canMerge(cur)) {
          prev.mergeBucket(cur);
          iterator.remove();
        } else {
          prev = cur;
        }
      } else {
        prev = cur;
      }
    }
  }

  /* Randomly reduces the number of buckets down to maxBuckets */
  private void reduceBuckets(ArrayList<ValueBucket> buckets, int maxBuckets, Random rand) {
    while(buckets.size() > maxBuckets) {
      int selectedToMerge = rand.nextInt(buckets.size()-1);
      buckets.get(selectedToMerge).mergeBucket(buckets.get(selectedToMerge+1));
      buckets.remove(selectedToMerge+1);
    }
  }

  /* Returns the ith candidate double */
  public double getCandidate(int index) {
    double endVal = buckets.get(index).maxValue;
    double startVal = buckets.get(index+1).minValue;
    return 0.5*(endVal+startVal);
  }

  /* Returns the total number of candidates */
  public int numCandidates() {
    return buckets.size()-1;
  }

  /* Returns an array A such that:
   *    for all c, A[c] is that number of instances from class c that have a value of
   *    target attribute that is less than the candidate at the specified index but
   *    greater than the candidate at the prior index */
  public int[] getCandidatesClassFreqs(int index) {
    int[] classFreqs = new int[classIndexMap.size()];
    for(String key : buckets.get(index).keySet()) {
      classFreqs[classIndexMap.get(key)] += buckets.get(index).get(key);
    }
    return classFreqs;
  }

  /* Represents a range of values for a particular attribute */
  private static class ValueBucket extends HashMap<String, Integer> implements Comparable<ValueBucket> {
    private double minValue;
    private double maxValue;

    /* Constructor */
    private ValueBucket(double minValue, double maxValue) {
      super();
      this.minValue = minValue;
      this.maxValue = maxValue;
    }

    /* Accessor for minValue */
    public double getMinValue() {
      return minValue;
    }

    /* Accessor for maxValue */
    public double getMaxValue() {
      return maxValue;
    }

    /* Returns whether or not the specified other object is a ValueBucket with the
     * same minValue */
    @Override
    public boolean equals(Object other) {
      if(other==this) {
        return true;
      }
      if(!(other instanceof ValueBucket)) {
        return false;
      }
      ValueBucket otherBucket = (ValueBucket)other;
      return this.minValue == otherBucket.minValue;
    }

    /* Returns a nicely formatted string reprsentation of the bucket */
    @Override
    public String toString() {
      return String.format("[%.2f-%.2f]: %s", minValue, maxValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(minValue);
    }

    /* Compares this ValueBucket to the specified other ValueBucket*/
    public int compareTo(ValueBucket other) {
      return ((Double)minValue).compareTo((Double)other.minValue);
    }

    /* Returns whether or not this bucket can successfully be merged with the
     * specified other bucket. Two buckets can be merged if they are both homogenous
     * with respect to the same class */
    private boolean canMerge(ValueBucket other) {
      if(!(other.size() <= 1 && size() <= 1)) {
        return false;
      }
      return keySet().containsAll(other.keySet()) || other.keySet().containsAll(keySet());
    }

    /* Merges the specified bucket into this bucket. */
    private void mergeBucket(ValueBucket other) {
      this.minValue = Math.min(minValue, other.minValue);
      this.maxValue = Math.max(maxValue, other.maxValue);
      for(String classLabel : other.keySet()) {
        int freq = other.get(classLabel);
        int prevFreq = getOrDefault(classLabel, 0);
        put(classLabel, freq+prevFreq);
      }
    }
  }
}

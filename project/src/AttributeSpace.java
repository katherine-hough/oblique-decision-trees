import java.util.PriorityQueue;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Objects;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Collections;

public class AttributeSpace {

  public static ArrayList<Double> getSplitBuckets(Iterable<Record> records, int attribute, double defaultValue, int maxBuckets) {
    HashMap<Double, ValueBucket> temp = new HashMap<>();
    for(Record record : records) {
      double value = record.getOrDefault(attribute, defaultValue);
      temp.putIfAbsent(value, new ValueBucket(value, value));
      ValueBucket bucket = temp.get(value);
      int prevFreq = bucket.getOrDefault(record.getClassLabel(), 0);
      bucket.put(record.getClassLabel(), prevFreq+1);
    }
    if(temp.size() <= 1) {
      return new ArrayList<Double>(); // all records have the same value for this attribute
    }
    ArrayList<ValueBucket> values = new ArrayList<>(temp.values());
    Collections.sort(values);
    Iterator<ValueBucket> iterator = values.iterator();
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
    int numBuckets = Math.min(maxBuckets, values.size()-1);
    ArrayList<Double> buckets = new ArrayList<>(numBuckets);
    int step = values.size()/(numBuckets+1);
    for(int i = step; i < values.size(); i+=step) {
      double endVal = values.get(i-1).maxValue;
      double startVal = values.get(i).minValue;
      buckets.add(0.5*(endVal+startVal));
    }
    return buckets;
  }

  /* Represents a range of values for a particular attribute */
  static class ValueBucket extends HashMap<String, Integer> implements Comparable<ValueBucket> {
    double minValue;
    double maxValue;

    /* Constructor */
    ValueBucket(double minValue, double maxValue) {
      super();
      this.minValue = minValue;
      this.maxValue = maxValue;
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
      return String.format("[%.2f-%.2f]: %s", minValue, maxValue, super.toString());
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
    boolean canMerge(ValueBucket other) {
      if(!(other.size() <= 1 && size() <= 1)) {
        return false;
      }
      return keySet().containsAll(other.keySet()) || other.keySet().containsAll(keySet());
    }

    /* Merges the specified bucket into this bucket. */
    void mergeBucket(ValueBucket other) {
      if(!canMerge(other)) {
        throw new RuntimeException("Invalid ValueBucket merge attempted.");
      }
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

import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

/* Represents a data instance with a sparsely represention of its features and
 * a class label. Maps feature numbers to values */
public class Record extends HashMap<Integer, Double> {

  private static final int MAX_BUCKETS = 100;
  public static final double DEFAULT_FEATURE_VALUE = 0.0;
  private final String classLabel; // the class of this record, null if no class

  /* Constructor. Initializes map based on the specified map. */
  public Record(String classLabel, HashMap<Integer, Double> features) {
    super(features);
    this.classLabel = classLabel;
  }

  /* Constructor. Initializes as an empty map. */
  public Record(String classLabel) {
    super();
    this.classLabel = classLabel;
  }

  /* Returns a string represention of the record */
  @Override
  public String toString() {
    return (classLabel==null ? "?" : classLabel) + ": " + size();
  }

  /* Converts the record into a dense string representation of all of its features */
  public String toDenseString(TreeSet<Integer> allFeatures) {
    String result = "";
    for(int feature : allFeatures) {
      result += getOrDefault(feature, DEFAULT_FEATURE_VALUE) + " ";
    }
    return result + classLabel;
  }

  /* Accessor for classLabel */
  public String getClassLabel() {
    return classLabel;
  }

  /* Returns record instances read in from the specified file with labels assigned
   * based on the contents of the second specified file. If sparse is true then
   * it is assumed that the records are represented sparsely in the first specified
   * file, otherwise a dense reprsentation is assumed. For the sparse represention
   * it is assumed that feature numbers come before feature weights */
  public static ArrayList<Record> readRecords(String trainingFile, String trainingLabelFile, boolean sparse) {
    ArrayList<String> labels = DataMiningUtil.readLines(trainingLabelFile);
    ArrayList<String> lines = DataMiningUtil.readLines(trainingFile);
    boolean missing = false;
    if(lines.size() != labels.size()) {
      throw new RuntimeException("Number of labels does not equal the number of vectors.");
    }
    ArrayList<Record> records = new ArrayList<>(lines.size());
    for(int i = 0; i < lines.size(); i++) {
      Record record = new Record(labels.get(i));
      if(addFeaturesToRecord(record, sparse, lines.get(i))) {
        missing = true;
      }
      records.add(record);
    }
    if(missing) {
      fixMissingAttributes(records);
    }
    return records;
  }

  /* Returns record instances read in from the specified file. If sparse is true then
   * it is assumed that the records are represented sparsely in the first specified
   * file, otherwise a dense reprsentation is assumed. For the sparse represention
   * it is assumed that feature numbers come before feature weights */
  public static ArrayList<Record> readRecords(String trainingFile, boolean sparse) {
    ArrayList<String> lines = DataMiningUtil.readLines(trainingFile);
    ArrayList<Record> records = new ArrayList<>(lines.size());
    boolean missing = false;
    for(int i = 0; i < lines.size(); i++) {
      Record record = new Record(null);
      if(addFeaturesToRecord(record, sparse, lines.get(i))) {
        missing = true;
      }
      records.add(record);
    }
    if(missing) {
      fixMissingAttributes(records);
    }
    return records;
  }

  /* Replaces missing attributes (represented with nulls with the median value
   * for the attribute in the record's class */
  private static void fixMissingAttributes(ArrayList<Record> records) {
    HashSet<Integer> allFeats = getAllFeatures(records);
    for(int feature: allFeats) {
      HashMap<String, ArrayList<Double>> classValuesMap = new HashMap<>();
      ArrayList<Double> values = new ArrayList<>();
      for(Record record : records) {
        if(record.containsKey(feature) && record.get(feature)!=null) {
          String label = record.getClassLabel();
          classValuesMap.putIfAbsent(label, new ArrayList<Double>());
          classValuesMap.get(label).add(record.get(feature));
          values.add(record.get(feature));
        }
      }
      double overallMedian = values.size()> 0 ? DataMiningUtil.getMedian(values) : -1;
      HashMap<String, Double> medians = new HashMap<>();
      for(String key : classValuesMap.keySet()) {
        medians.put(key, DataMiningUtil.getMedian(classValuesMap.get(key)));
      }
      for(Record record : records) {
        if(record.containsKey(feature) && record.get(feature)==null) {
          String label = record.getClassLabel();
          if(medians.containsKey(label)) {
            // use the median for this record's class
            record.put(feature, medians.get(label));
          } else if (overallMedian > 0) {
            // no other records in the same class contained this feature
            // use the overall median
            record.put(feature, overallMedian);
          } else {
            // no other records contained this feature
            // remove the feature
            record.remove(feature);
          }
        }
      }
    }
  }

  /* Adds features extracts from the specified line to the specified record. Returns
   * whether a missing attribute was found */
  private static boolean addFeaturesToRecord(Record record, boolean sparse, String line) {
    boolean missing = false;
    line = line.replaceAll("^\\s+",""); // remove leading whitespace
    String[] temp = line.split("\\s+"); // split at whitespace
    if(sparse) {
      if(temp.length%2!=0) {
        throw new RuntimeException("Missing a weight for a feature in a sparsely represented feature vector.");
      }
      for(int j = 0; j < temp.length; j+=2) {
        double value = Double.parseDouble(temp[j+1]);
        if(value != DEFAULT_FEATURE_VALUE) {
          record.put(Integer.parseInt(temp[j]), value);
        }
      }
    } else {
      for(int j = 0; j < temp.length; j++) {
        try {
          double value = Double.parseDouble(temp[j]);
          if(value != DEFAULT_FEATURE_VALUE) {
            record.put(j, value);
          }
        } catch (NumberFormatException e) {
          record.put(j, null); // put null for missing attributes
          missing = true;
        }
      }
    }
    return missing;
  }

  /* Returns all the feature keys contained in the specified iterable of records */
  public static HashSet<Integer> getAllFeatures(Iterable<Record> records) {
    HashSet<Integer> features = new HashSet<>();
    for(Record record : records) {
      features.addAll(record.keySet());
    }
    return features;
  }

  /* Returns values to split the specified feature at */
  public static ArrayList<Double> getSplitBuckets(List<Record> records, int feature) {
    return AttributeSpace.getSplitBuckets(records, feature, DEFAULT_FEATURE_VALUE, MAX_BUCKETS);
  }
}

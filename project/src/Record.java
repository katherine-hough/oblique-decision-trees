import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/* Represents a data instance with a sparsely represention of its features and
 * a class label. Maps feature numbers to values */
public class Record extends HashMap<Integer, Double> {

  private static final int MAX_BUCKETS = 100;
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
    if(lines.size() != labels.size()) {
      throw new RuntimeException("Number of labels does not equal the number of vectors.");
    }
    ArrayList<Record> records = new ArrayList<>(lines.size());
    for(int i = 0; i < lines.size(); i++) {
      Record record = new Record(labels.get(i));
      addFeaturesToRecord(record, sparse, lines.get(i));
      records.add(record);
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
    for(int i = 0; i < lines.size(); i++) {
      Record record = new Record(null);
      addFeaturesToRecord(record, sparse, lines.get(i));
      records.add(record);
    }
    return records;
  }

  /* Adds features extracts from the specified line to the specified record */
  private static void addFeaturesToRecord(Record record, boolean sparse, String line) {
    line = line.replaceAll("^\\s+",""); // remove leading whitespace
    String[] temp = line.split("\\s+"); // split at whitespace
    if(sparse) {
      if(temp.length%2!=0) {
        throw new RuntimeException("Missing a weight for a feature in a sparsely represented feature vector.");
      }
      for(int j = 0; j < temp.length; j+=2) {
          record.put(Integer.parseInt(temp[j]), Double.parseDouble(temp[j+1]));
      }
    } else {
      for(int j = 0; j < temp.length; j++) {
          record.put(j, Double.parseDouble(temp[j]));
      }
    }
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
  public static HashSet<Double> getSplitBuckets(List<Record> records, int feature, double defaultValue) {
    int numBuckets = Math.min(records.size()-1, MAX_BUCKETS);
    HashSet<Double> buckets = new HashSet<>();
    ArrayList<Double> values= new ArrayList<>(records.size());
    for(Record record : records) {
      values.add(record.getOrDefault(feature, defaultValue));
    }
    Collections.sort(values);
    int step = records.size()/(numBuckets+1);
    for(int i = step; i < records.size(); i+=step) {
      buckets.add((records.get(i-1).get(feature)+records.get(i).get(feature))/2.0);
    }
    return buckets;
  }
}

import java.util.ArrayList;

/* Represents some method for calculating the classes of test instances */
public abstract class Classifier {

  /* Classifies a single training instance and returns a string representation of
   * that calculated class */
  public abstract String classify(Record record);

  /* Classifies every instances in the specified iterable of records and returns a
   * list of those calculated classes */
  public ArrayList<String> classifyAll(Iterable<Record> records) {
    ArrayList<String> classes = new ArrayList<>();
    for(Record record : records) {
      classes.add(classify(record));
    }
    return classes;
  }
}

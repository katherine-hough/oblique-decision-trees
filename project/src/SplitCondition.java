import java.util.function.Predicate;

/* Represents a condition to split the instances at a node of a decision tree at */
public class SplitCondition implements Comparable<SplitCondition> {

  private final String desc;
  private final Predicate<Record> condition;
  private double impurity;
  private int rank;
  private int feature; // optional field, feature this split occurs on
  private double bucket; // optional field, splitting value for feature

  public SplitCondition(Predicate<Record> condition, String desc) {
    this.condition = condition;
    this.desc = desc;
    this.rank = 1;
    this.impurity = -1;
  }

  public SplitCondition(Predicate<Record> condition, int feature, double bucket) {
    this.condition = condition;
    this.desc = String.format("x[%d]<%2.2f", feature, bucket);
    this.feature = feature;
    this.bucket = bucket;
    this.rank = 1;
    this.impurity = -1;
  }

  /* Accessor for feature */
  public int getFeature() {
    return feature;
  }

  /* Accessor for bucket */
  public double getBucket() {
    return bucket;
  }

  /* Returns a copy of this SplitCondition without its impurity set */
  public SplitCondition copy() {
    SplitCondition copy = new SplitCondition(this.condition, this.desc);
    copy.rank = this.rank;
    return copy;
  }

  /* Setter for impurity */
  public void setImpurity(double impurity) {
    this.impurity = impurity;
  }

  /* Getter for impurity */
  public double getImpurity() {
    return impurity;
  }

  /* Compares the specified other SplitCondition to this SplitCondition */
  public int compareTo(SplitCondition other) {
    int comp = ((Double)impurity).compareTo((Double)other.impurity);
    if(comp == 0) {
      return ((Integer)rank).compareTo((Integer)other.rank);
    } else {
      return comp;
    }
  }

  /* Returns whether the specified record passes the condition */
  public boolean test(Record record) {
    return condition.test(record);
  }

  /* Returns a string representation of the split condition */
  @Override
  public String toString() {
    return String.format("%s", desc);
  }

  /* Returns a condition that is the logical negation of this condition */
  public SplitCondition negate() {
    String newDesc = String.format("!(%s)", desc);
    SplitCondition split = new SplitCondition(condition.negate(), newDesc);
    split.rank = this.rank;
    return split;
  }

  /* Returns a condition that is the logical OR of this condition and the specified other condition */
  public SplitCondition or(SplitCondition other) {
    String newDesc = String.format("(%s)||(%s)", desc, other.desc);
    SplitCondition split = new SplitCondition(condition.or(other.condition), newDesc);
    split.rank = this.rank + other.rank;
    return split;
  }

  /* Returns a condition that is the logical AND of this condition and the specified other condition */
  public SplitCondition and(SplitCondition other) {
    String newDesc = String.format("(%s)&&(%s)", desc, other.desc);
    SplitCondition split = new SplitCondition(condition.and(other.condition), newDesc);
    split.rank = this.rank + other.rank;
    return split;
  }
}

import java.util.function.Predicate;

/* Represents a condition to split the instances at a node of a decision tree at */
public class SplitCondition implements Comparable<SplitCondition> {

  private final String desc;
  private final Predicate<Record> condition;
  private double impurity;

  public SplitCondition(String desc, Predicate<Record> condition) {
    this.desc = desc;
    this.condition = condition;
  }

  /* Setter for impurity */
  public void setImpurity(double impurity) {
    this.impurity = impurity;
  }

  /* Compares the specified other SplitCondition to this SplitCondition */
  public int compareTo(SplitCondition other) {
    return ((Double)impurity).compareTo((Double)other.impurity);
  }

  /* Returns whether the specified record passes the condition */
  public boolean test(Record record) {
    return condition.test(record);
  }

  /* Returns a string representation of the split condition */
  @Override
  public String toString() {
    return desc;
  }

  /* Returns a condition that is the logical negation of this condition */
  public SplitCondition negate() {
    String newDesc = String.format("NOT(%s)", desc);
    return new SplitCondition(newDesc, condition.negate());
  }

  /* Returns a condition that is the logical OR of this condition and the specified other condition */
  public SplitCondition or(SplitCondition other) {
    String newDesc = String.format("(%s) OR (%s)", desc, other.desc);
    return new SplitCondition(newDesc, condition.or(other.condition));
  }

  /* Returns a condition that is the logical AND of this condition and the specified other condition */
  public SplitCondition and(SplitCondition other) {
    String newDesc = String.format("(%s) AND (%s)", desc, other.desc);
    return new SplitCondition(newDesc, condition.and(other.condition));
  }
}

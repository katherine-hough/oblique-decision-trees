import java.util.function.Predicate;

/* Represents a condition to split the instances at a node of a decision tree at */
public class SplitCondition implements Comparable<SplitCondition> {

  private final String desc;
  private final Predicate<Record> condition;
  private double impurity;
  private int rank;

  public SplitCondition(String desc, Predicate<Record> condition) {
    this.desc = desc;
    this.condition = condition;
    this.rank = 1;
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
    double rankFactor = 1 + 0.1*(rank-1);
    double otherRankFactor = 1 + 0.1*(other.rank-1);
    int comp = ((Double)(impurity*rankFactor)).compareTo((Double)(other.impurity*otherRankFactor));
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
    return String.format("%s | Impurity: %3.3f | Rank: %d", desc, impurity, rank);
  }

  /* Returns a condition that is the logical negation of this condition */
  public SplitCondition negate() {
    String newDesc = String.format("NOT(%s)", desc);
    SplitCondition split = new SplitCondition(newDesc, condition.negate());
    split.rank = this.rank;
    return split;
  }

  /* Returns a condition that is the logical OR of this condition and the specified other condition */
  public SplitCondition or(SplitCondition other) {
    String newDesc = String.format("(%s) OR (%s)", desc, other.desc);
    SplitCondition split = new SplitCondition(newDesc, condition.or(other.condition));
    split.rank = this.rank + other.rank;
    return split;
  }

  /* Returns a condition that is the logical AND of this condition and the specified other condition */
  public SplitCondition and(SplitCondition other) {
    String newDesc = String.format("(%s) AND (%s)", desc, other.desc);
    SplitCondition split = new SplitCondition(newDesc, condition.and(other.condition));
    split.rank = this.rank + other.rank;
    return split;
  }
}

import java.util.Objects;

public class Pair<K,V> {

  private K key;
  private V value;

  /* Constructor */
  public Pair(K key, V value) {
    this.key = key;
    this.value = value;
  }

  @Override
  /* Test this Pair for equality with another Object. */
  public boolean equals(Object other) {
    if(other == this) {
      return true;
    }
    if(!(other instanceof Pair)) {
      return false;
    }
    Pair pair = (Pair) other;
    return pair.key.equals(key) && pair.value.equals(value);
  }

  @Override
  public int hashCode() {
    return Objects.hash(key, value);
  }

  /* Getter for key */
  public K getKey() {
    return key;
  }

  /* Getter for value */
  public V getValue() {
    return value;
  }

  @Override
  public String toString() {
    return "Key: " + key + " | Value: " + value;
  }
}

import java.io.PrintWriter;
import java.io.IOException;
import java.io.File;
import java.util.Scanner;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Function;
import java.util.Collection;
import java.util.Random;
import java.util.Collections;
import java.util.List;

/* Utility Class that contains methods that help with solving data mining problems */
public class DataMiningUtil {

  /* Returns a list of the String lines of the specified file */
  public static ArrayList<String> readLines(String filename) {
    try {
      Scanner sc = new Scanner(new File(filename));
      ArrayList<String> lines = new ArrayList<>();
      while(sc.hasNextLine()) {
        lines.add(sc.nextLine());
      }
      sc.close();
      return lines;
    } catch (IOException e) {
      throw new RuntimeException("Error occurred reading from file: " + filename);
    }
  }

  /* Writes the specified items to the specified file */
  public static <T> void writeToFile(Iterable<T> items, String filename) {
    try {
      PrintWriter pw = new PrintWriter(filename);
      for(T item : items) {
        pw.println(item.toString());
      }
      pw.close();
    }
    catch (IOException e) {
      System.err.println("Error occurred writing to file: " + filename);
    }
  }

  /* Writes the specified items to the specified file */
  public static <T> void writeToFile(T[] items, String filename) {
    try {
      PrintWriter pw = new PrintWriter(filename);
      for(T item : items) {
        pw.println(item.toString());
      }
      pw.close();
    }
    catch (IOException e) {
      System.err.println("Error occurred writing to file: " + filename);
    }
  }

  /* Calculates the mean or average value for the iterable of numbers */
  public static double mean(Iterable<? extends Number> values) {
    double sum = 0;
    int count = 0;
    for(Number value : values) {
      sum += value.doubleValue();
      count++;
    }
    return sum/count;
  }

  /* Calculates sample standard deviation for the iterable of numbers */
  public static double sampleStandardDeviation(Iterable<? extends Number> values) {
    double mean = mean(values);
    int n = 0;
    double sum = 0;
    for(Number value : values) {
      sum += Math.pow(value.doubleValue()-mean, 2);
      n++;
    }
    return Math.sqrt(sum * 1.0/(n-1));
  }

  /* Calculates population standard deviation for the iterable of numbers */
  public static double populationStandardDeviation(Iterable<? extends Number> values) {
    double mean = mean(values);
    int n = 0;
    double sum = 0;
    for(Number value : values) {
      sum += Math.pow(value.doubleValue()-mean, 2);
      n++;
    }
    return Math.sqrt((sum * 1.0)/n);
  }

  /* Returns a HashMap representing the frequency of "events" in the iterable.
   * Iterates over the elements in the specified Iterable of elements
   * and for each element applies the specified function to get a iterable of "events" */
  public static <T, U> HashMap<U, Integer> createFreqMap(Iterable<T> elements, Function<T, Iterable<U>> function) {
    HashMap<U, Integer> freqMap = new HashMap<>();
    for(T element : elements) {
      Iterable<U> events = function.apply(element);
      for(U event : events) {
        freqMap.put(event, freqMap.getOrDefault(event, 0)+1);
      }
    }
    return freqMap;
  }

  /* Returns a HashMap representing the frequency of "events" in the iterable.
   * Iterates over the elements in the specified Iterable of elements
   * and for each element applies the specified function to get a iterable of "events" */
  public static <T, U> HashMap<U, Double> createDoubleFreqMap(Iterable<T> elements, Function<T, Iterable<U>> function) {
    HashMap<U, Double> freqMap = new HashMap<>();
    for(T element : elements) {
      Iterable<U> events = function.apply(element);
      for(U event : events) {
        freqMap.put(event, freqMap.getOrDefault(event, 0.0)+1.0);
      }
    }
    return freqMap;
  }

  /* Returns the specified numGroups number of random groups of Strings that partition
   * spacified iterable of Strings. This groups are stratified with respect to
   * specified list of classes.
   * Uses the specified Random object for shuffling to produce randomness.*/
  public static ArrayList<ArrayList<String>> getStratifiedGroups(Iterable<String> instances, int numGroups, List<String> classes, Random rand) {
    ArrayList<ArrayList<String>> groups = new ArrayList<>(numGroups); // the created groups to be returned
    for(int i = 0; i < numGroups; i++) {
      groups.add(new ArrayList<String>());
    }
    HashMap<String, ArrayList<String>> subpops = new HashMap<>();
    int x = 0;
    for(String instance : instances) {
      subpops.putIfAbsent(classes.get(x), new ArrayList<String>());
      subpops.get(classes.get(x++)).add(instance);
    }
    for(String key : subpops.keySet()) {
      Collections.shuffle(subpops.get(key), rand);
      for(int i = 0; i < subpops.get(key).size(); i++) {
        groups.get(i%numGroups).add(subpops.get(key).get(i));
      }
    }
    return groups;
  }

  /* Returns the specified numGroups number of random groups of Strings that partition
   * spacified iterable of Strings. This groups are stratified with respect to
   * the string in the specified column  (zero-indexed, when the instance is split on whitespace).
   * Uses the specified Random object for shuffling to produce randomness. */
  public static ArrayList<ArrayList<String>> getStratifiedGroups(Iterable<String> instances, int numGroups, int targetCol, Random rand) {
    ArrayList<String> classes = new ArrayList<>();
    for(String instance : instances) {
      String line = instance.replaceAll("^\\s+",""); // remove leading whitespace
      String[] temp = line.split("\\s+"); // split at whitespace
      classes.add(temp[targetCol]);
    }
    return getStratifiedGroups(instances, numGroups, classes, rand);
  }
}

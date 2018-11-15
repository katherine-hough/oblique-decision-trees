import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.Collection;
import java.util.TreeSet;
import java.util.Random;

public class GeneticAlgorithmSplitter {
  // Selectiveness of the tournament selection
  private static final int TOURNAMENT_SIZE = 7;
  // The records being considered when making this split
  List<Record> records;
  // Maps each different class label found for a record to a different integer index
  HashMap<String, Integer> classIndexMap;
  // The features which are considered in the split in the order they are represented
  // in individual's genes
  int[] targetFeatures;
  // Size of population
  int populationSize;
  // Random number generator
  Random rand;
  // Maximum number of buckets considered for splitting per attribute
  int maxBuckets;

  public GeneticAlgorithmSplitter(List<Record> records, int[] targetFeatures, int populationSize, Random rand, int maxBuckets) {
    if(populationSize%2!=0) {
      throw new RuntimeException("Population size must be even.");
    }
    this.records = records;
    this.classIndexMap = new HashMap<>();
    for(Record record : records) {
      classIndexMap.putIfAbsent(record.getClassLabel(), classIndexMap.size());
    }
    this.targetFeatures = targetFeatures;
    this.populationSize = populationSize;
    this.rand = rand;
    this.maxBuckets = maxBuckets;
  }

  /* Return the best split condition found after the specified maximum number of
   * generations */
  public SplitCondition getBestSplitCondition(int maxGenerations) {
    Individual[] population = initializePopulation();
    if(population == null) {
      return null;
    }
    double[][] valueRanges = getValueRanges();
    Individual best = null;
    for(int gen = 0; gen < maxGenerations; gen++) {
      Individual popBest = assessFitness(population);
      best = (best == null || popBest.fitness > best.fitness) ? popBest : best;
      Individual[] nextGen = new Individual[populationSize];
      for(int c = 0; c < populationSize; c+=2) {
        Individual parent1 = selectParent(population);
        Individual parent2 = selectParent(population);
        // Individual[] children = uniformCrossover(parent1, parent2);
        Individual[] children = intermediateRecombination(parent1, parent2, valueRanges);
        mutate(children[0], valueRanges);
        mutate(children[1], valueRanges);
        nextGen[c] = children[0];
        nextGen[c+1] = children[1];
      }
      population = nextGen;
    }
    return best!= null ? best.toSplitCondition() : null;
  }

  /* Returns arrays of the min and max values for each target attribute */
  private double[][] getValueRanges() {
    double[][] ranges = new double[2][targetFeatures.length+1];
    for(int i = 0; i < ranges[0].length-1; i++) {
      int featureNum = targetFeatures[i];
      ranges[0][i] = 0.0;
      ranges[1][i] = 0.0;
      for(Record record : records) {
        double value = 1.0/record.getOrDefault(featureNum);
        ranges[0][i] = Math.min(ranges[0][i], 0.5*value);
        ranges[1][i] = Math.max(ranges[1][i], 2*value);
      }
    }
    ranges[0][ranges[0].length-1] = 0;
    ranges[1][ranges[1].length-1] = targetFeatures.length;
    return ranges;
  }

  /* Mutates the childs genes using Gaussian Convolution */
  private void mutate(Individual child, double[][] valueRanges) {
    double noiseProb = 1.0;
    double std = 0.1;
    for(int i = 0; i < child.genes.length; i++) {
      if(noiseProb >= rand.nextDouble()) {
        double n;
        do {
          n = rand.nextGaussian()*std;
        } while(!inRange(child.genes[i]+n, valueRanges[0][i], valueRanges[1][i]));
        child.genes[i]+=n;
      }
    }
  }

  /* Produces child instances from the specified parents using intermediate recombination */
  private Individual[] intermediateRecombination(Individual parent1, Individual parent2, double[][] valueRanges) {
    double p = 0.25;
    Individual[] children = new Individual[2];
    children[0] = new Individual();
    children[1] = new Individual();
    for(int i = 0; i < parent1.genes.length; i++) {
      double x;
      double y;
      do {
        double a = (2*p+1)*rand.nextDouble()-p;
        double b = (2*p+1)*rand.nextDouble()-p;
        x = a*parent1.genes[i] + (1-a)*parent2.genes[i];
        y = b*parent2.genes[i] + (1-b)*parent1.genes[i];
      } while(!(inRange(x, valueRanges[0][i], valueRanges[1][i]) && inRange(y, valueRanges[0][i], valueRanges[1][i])));
      children[0].genes[i] = x;
      children[1].genes[i] = y;
    }
    return children;
  }

  /* Returns whether or not the specified value is in the specified inclusive range */
  private boolean inRange(double value, double min, double max) {
    return value >= min && value <= max;
  }

  /* Swaps the values at the same index in each parent at random to produce child instances */
  private Individual[] uniformCrossover(Individual parent1, Individual parent2) {
    double swapProb = 1/parent1.genes.length;
    Individual[] children = new Individual[2];
    children[0] = new Individual();
    children[1] = new Individual();
    for(int i = 0; i < parent1.genes.length; i++) {
      if(swapProb >= rand.nextDouble()) {
        children[0].genes[i] = parent2.genes[i];
        children[1].genes[i] = parent1.genes[i];
      } else {
        children[0].genes[i] = parent1.genes[i];
        children[1].genes[i] = parent2.genes[i];
      }
    }
    return children;
  }

  /* Selects a parent from the population using tournament selection */
  private Individual selectParent(Individual[] population) {
    Individual best = population[rand.nextInt(population.length)];
    for(int i = 2; i <= TOURNAMENT_SIZE; i++) {
      Individual next = population[rand.nextInt(population.length)];
      if(next.fitness > best.fitness) {
        best = next;
      }
    }
    return best;
  }

  /* Initializes the population */
  private Individual[] initializePopulation() {
    Individual[] population = new Individual[populationSize];
    for(int p = 0; p < population.length; p++) {
      population[p] = new Individual();
    }
    for(int i = 0; i < targetFeatures.length; i++) {
      ArrayList<Double> buckets = AttributeSpace.getSplitBuckets(records, targetFeatures[i], maxBuckets);
      if(buckets.size() == 0) {
        return null;
      }
      for(int p = 0; p < population.length; p++) {
        if(rand.nextInt(2) == 0) {
          population[p].genes[i] = 1.0/buckets.get(rand.nextInt(buckets.size()));
          population[p].genes[population[p].genes.length-1]++;
        }
      }
    }
    return population;
  }

  /* Computes the fitness of each member of the population and returns the most
   * fit individual in the population */
  private Individual assessFitness(Individual[] population) {
    Individual best = null;
    for(Individual member : population) {
      double fitness = member.calcFitness();
      if(best == null || best.fitness < fitness) {
        best = member;
      }
    }
    return best;
  }

  /* The weighted GINI impurity if the records are partitioned based on the
   * specified condition */
  private double getTotalGiniImpurity(SplitCondition splitCondition) {
    int[] classFreqsLeft = new int[classIndexMap.size()];
    int[] classFreqsRight = new int[classIndexMap.size()];
    int totalLeft = 0;
    int totalRight = 0;
    for(Record record : records) {
      int index = classIndexMap.get(record.getClassLabel());
      if(splitCondition.test(record)) {
        totalLeft++;
        classFreqsLeft[index]++;
      } else {
        totalRight++;
        classFreqsRight[index]++;
      }
    }
    double giniLeft = getGiniImpurity(classFreqsLeft);
    double giniRight = getGiniImpurity(classFreqsRight);
    double probLeft = (1.0*totalLeft)/records.size();
    double probRight = (1.0*totalRight)/records.size();
    return giniLeft*probLeft + giniRight*probRight;
  }

  /* Calculates the GINI impurity based on the specified array of class frequencies*/
  public static double getGiniImpurity(int[] classFreqs) {
    int sum = 0;
    int total = 0;
    for(int classFreq : classFreqs) {
      sum += classFreq*classFreq;
      total += classFreq;
    }
    return 1.0 - sum/(1.0 * total * total);
  }

  /* Represents an individual in the population, encode a splits condition for the
   * records */
  class Individual {
    double[] genes;
    double fitness;

    /* Constructor */
    Individual() {
      genes = new double[targetFeatures.length+1];
    }

    /* Returns a string representation of the individual */
    @Override
    public String toString() {
      String desc = "";
      for(int i = 0; i < genes.length-1; i++) {
        if(genes[i] != 0) {
          desc += String.format("%3.3f*X_%d + ", genes[i], targetFeatures[i]);
        }
      }
      desc += String.format("-%3.3f < 0", genes[genes.length-1]);
      return desc;
    }

    /* Calculates and stores the fitness of the individual. Returns the calculated
     * value */
    double calcFitness() {
      this.fitness = 1 - getTotalGiniImpurity(this.toSplitCondition());
      return this.fitness;
    }

    /* Decodes the genes into a SplitCondition for the records */
    SplitCondition toSplitCondition() {
      Predicate<Record> pred = (record) -> {
        double sum = 0;
        for(int i = 0; i < genes.length-1; i++) {
          sum += genes[i] * record.getOrDefault(targetFeatures[i]);
        }
        return sum - genes[genes.length-1] < 0;
      };
      return new SplitCondition(toString(), pred);
    }
  }
}

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.function.Predicate;
import java.util.Random;
import java.util.PriorityQueue;

/* Uses a genetic algorithm to determine the oblique split to be made on a list of
 * records */
public class GeneticSplitter {

  /* The records being considered when making this split */
  private final List<Record> records;
  /* Maps each different class label to a different integer index */
  private final HashMap<String, Integer> classIndexMap;
  /* The features which are considered in the split in the order they are represented
   * in individual's genes */
  private final int[] targetFeatures;
  /* Random number generator */
  private final Random rand;
  /* Size of population */
  private final int populationSize;
  /* Maximum number of buckets considered for splitting per attribute */
  private final int maxBuckets;
  /* Selectiveness of the tournament selection */
  private final int tournamentSize;
  /* Selectiveness of the tournament selection for population replacement */
  private final int replacementTournamentSize;
  /* Maximum number of generations */
  private final int maxGenerations;
  /* Top conditions used to initialize the population */
  private final List<SplitCondition> topConditions;

  /* Private constructor called by the builder */
  private GeneticSplitter(GeneticSplitterBuilder builder) {
    this.records = builder.records;
    this.classIndexMap = builder.classIndexMap;
    this.targetFeatures = builder.targetFeatures;
    this.rand = builder.rand;
    this.populationSize = builder.populationSize;
    this.maxBuckets = builder.maxBuckets;
    this.tournamentSize = builder.tournamentSize;
    this.replacementTournamentSize = builder.replacementTournamentSize;
    this.maxGenerations = builder.maxGenerations;
    this.topConditions = builder.topConditions;
  }

  /* Return the best split condition found after the maximum number of
   * generations */
  public SplitCondition getBestSplitCondition() {
    if(targetFeatures.length == 0) {
      return null;
    } else if(targetFeatures.length == 1) {
      return topConditions.get(0);
    }
    Individual[] population = initializePopulation();
    if(population == null) {
      return null;
    }
    Individual best = null;
    for(Individual member : population) {
      if(best == null || member.updateFitness() > best.fitness) {
        best = member;
      }
    }
    double prevAvgFitness = -1;
    for(int gen = 0; gen < maxGenerations; gen++) {
      double curAvgFitness = getAverageFitness(population);
      if(curAvgFitness <= prevAvgFitness && gen >= 3) {
        return best.toSplitCondition();
      }
      prevAvgFitness = curAvgFitness;
      for(int c = 0; c < populationSize; c+=2) {
        if(best != null && best.fitness == 1.0) {
          // Optimal split was found
          return best.toSplitCondition();
        }
        Individual parent1 = selectParent(population);
        Individual parent2 = selectParent(population);
        Individual[] children = intermediateRecombination(parent1, parent2);
        mutate(children[0]);
        mutate(children[1]);
        if(best == null || children[0].updateFitness() > best.fitness) {
          best = children[0];
        }
        if(best == null || children[1].updateFitness() > best.fitness) {
          best = children[1];
        }
        replaceMembers(population, children);
      }
    }
    return best!= null ? best.toSplitCondition() : null;
  }

  /* Returns the average fitness of a member of the population */
  private double getAverageFitness(Individual[] population) {
    double sum = 0;
    for(Individual member : population) {
      sum += member.fitness;
    }
    return sum/population.length;
  }

  /* Selects to members of the specified population and replaces them with the
   * specified children */
  private void replaceMembers(Individual[] population, Individual[] children) {
    int worstIndex1 = rand.nextInt(population.length);
    for(int i = 2; i <= replacementTournamentSize; i++) {
      int nextIndex = rand.nextInt(population.length);
      if(population[nextIndex].fitness < population[worstIndex1].fitness) {
        worstIndex1 = nextIndex;
      }
    }
    int worstIndex2;
    do {
      worstIndex2 = rand.nextInt(population.length);
    } while(worstIndex1 == worstIndex2);
    for(int i = 2; i <= replacementTournamentSize; i++) {
      int nextIndex;
      do {
        nextIndex = rand.nextInt(population.length);
      } while(nextIndex == worstIndex1);
      if(population[nextIndex].fitness < population[worstIndex2].fitness) {
        worstIndex2 = nextIndex;
      }
    }
    population[worstIndex1] = children[0];
    population[worstIndex2] = children[1];
  }

  /* Mutates the childs genes using Gaussian Convolution */
  private void mutate(Individual child) {
    double noiseProb = 1.0;
    double std = 0.1;
    for(int i = 0; i < child.genes.length; i++) {
      if(noiseProb >= rand.nextDouble()) {
        child.genes[i]+=rand.nextGaussian()*std;
      }
    }
  }

  /* Produces child instances from the specified parents using intermediate recombination */
  private Individual[] intermediateRecombination(Individual parent1, Individual parent2) {
    double p = 0.25;
    Individual[] children = new Individual[2];
    children[0] = new Individual();
    children[1] = new Individual();
    for(int i = 0; i < parent1.genes.length; i++) {
      double a = (2*p+1)*rand.nextDouble()-p;
      double b = (2*p+1)*rand.nextDouble()-p;
      double x = a*parent1.genes[i] + (1-a)*parent2.genes[i];
      double y = b*parent2.genes[i] + (1-b)*parent1.genes[i];
      children[0].genes[i] = x;
      children[1].genes[i] = y;
    }
    return children;
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
    for(int i = 2; i <= tournamentSize; i++) {
      Individual next = population[rand.nextInt(population.length)];
      if(next.fitness > best.fitness) {
        best = next;
      }
    }
    return best;
  }

  /* Initializes the population. */
  private Individual[] initializePopulation() {
    Individual[] population = new Individual[populationSize];
    int lastGene = targetFeatures.length;
    for(int p = 0; p < population.length; p++) {
      population[p] = new Individual();
      population[p].genes[rand.nextInt(targetFeatures.length)] = 1.0;
    }
    HashMap<Integer, List<Double>> featureBucketsMap = new HashMap<>();
    for(SplitCondition condition : topConditions) {
      featureBucketsMap.putIfAbsent(condition.getFeature(), new ArrayList<>());
      featureBucketsMap.get(condition.getFeature()).add(condition.getBucket());
    }
    for(int i = 0; i < targetFeatures.length; i++) {
      int selected = rand.nextInt(population.length);
      population[selected].genes[i] = 1.0;
      List<Double> buckets = featureBucketsMap.get(targetFeatures[i]);
      for(int p = 0; p < population.length; p++) {
        if(rand.nextInt(buckets.size()) != 0) {
          population[p].genes[i] = 1.0;
        }
        if(population[p].genes[i] != 0) {
          population[p].genes[lastGene]+= buckets.get(rand.nextInt(buckets.size()));
        }
      }
    }
    return population;
  }

  /* Represents an individual in the population, encode a splits condition for the
   * records */
  private class Individual implements Comparable<Individual> {
    double[] genes;
    double fitness;

    /* Constructor */
    Individual() {
      this.genes = new double[targetFeatures.length+1];
    }

    /* Recalculates the individual's fitness */
    double updateFitness() {
      this.fitness = 1 - SplitStrategy.getTotalGiniImpurity(records, this.toSplitCondition(), classIndexMap);
      return this.fitness;
    }

    /* Compares this Individual to the specified oter Individual */
    public int compareTo(Individual other) {
      return ((Double)fitness).compareTo((Double)other.fitness);
    }

    /* Returns a string representation of the individual */
    @Override
    public String toString() {
      String desc = "";
      for(int i = 0; i < genes.length-1; i++) {
        if(genes[i] != 0) {
          desc += String.format("%.5f x[%d] + ", genes[i], targetFeatures[i]);
        }
      }
      desc += String.format("%.5f < 0", -1 * genes[genes.length-1]);
      return desc;
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
      SplitCondition cond = new SplitCondition(pred, toString());
      cond.setImpurity(1-fitness);
      return cond;
    }
  }

  /* Nested builder class for creating GeneticSplitters */
  public static class GeneticSplitterBuilder {
    private List<Record> records;
    private HashMap<String, Integer> classIndexMap;
    private int[] targetFeatures;
    private Random rand;
    private int populationSize;
    private int maxBuckets;
    private int tournamentSize;
    private int replacementTournamentSize;
    private int maxGenerations;
    private List<SplitCondition> topConditions;

    public GeneticSplitterBuilder records(List<Record> records) {
      this.records = records;
      return this;
    }

    public GeneticSplitterBuilder classIndexMap(HashMap<String, Integer> classIndexMap) {
      this.classIndexMap = classIndexMap;
      return this;
    }

    public GeneticSplitterBuilder targetFeatures(int[] targetFeatures) {
      this.targetFeatures = targetFeatures;
      return this;
    }

    public GeneticSplitterBuilder rand(Random rand) {
      this.rand = rand;
      return this;
    }

    public GeneticSplitterBuilder populationSize(int populationSize) {
      if(populationSize%2 != 0) {
        throw new RuntimeException("Genetic algorithm population size must be even.");
      }
      this.populationSize = populationSize;
      return this;
    }

    public GeneticSplitterBuilder maxBuckets(int maxBuckets) {
      this.maxBuckets = maxBuckets;
      return this;
    }

    public GeneticSplitterBuilder tournamentSize(int tournamentSize) {
      this.tournamentSize = tournamentSize;
      return this;
    }

    public GeneticSplitterBuilder replacementTournamentSize(int replacementTournamentSize) {
      this.replacementTournamentSize = replacementTournamentSize;
      return this;
    }

    public GeneticSplitterBuilder maxGenerations(int maxGenerations) {
      this.maxGenerations = maxGenerations;
      return this;
    }

    public GeneticSplitterBuilder topConditions(List<SplitCondition> topConditions) {
      this.topConditions = topConditions;
      return this;
    }

    /* Returns a GeneticSplitter instance built from the builders parameters */
    public GeneticSplitter build() {
      return new GeneticSplitter(this);
    }
  }
}

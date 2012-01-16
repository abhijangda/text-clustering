import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Solution for Quora Classifier question from CodeSprint 2012. This class
 * implements a logistic regression classifier trained with Stochastic Gradient
 * Descent. No regularization is used as it wasn't necessary for this problem
 */
public class QuoraClassifier {

  /**
   * A helper class representing a vector (an array of doubles)
   */
  private class Vector {

    private final double[] elements;

    public Vector(int numElements) {
      elements = new double[numElements];
    }

    public Vector clone() {
      Vector vClone = new Vector(this.getNumElements());
      vClone.set(this);
      return vClone;
    }

    public int getNumElements() {
      return elements.length;
    }

    public void add(Vector v) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] += v.get(i);
      }
    }

    public void subtract(Vector v) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] -= v.get(i);
      }
    }

    public void multiply(double multiplier) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] *= multiplier;
      }
    }

    public void multiply(Vector v) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] *= v.get(i);
      }
    }

    public void divide(double divisor) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] /= divisor;
      }
    }

    public void divide(Vector v) {
      for (int i = 0; i < elements.length; i++) {
        double divisor = v.get(i);
        if (divisor != 0.0) {
          elements[i] /= divisor;
        }
      }
    }

    public void elementwiseSquare() {
      for (int i = 0; i < elements.length; i++) {
        elements[i] *= elements[i];
      }
    }

    public void sqrt() {
      for (int i = 0; i < elements.length; i++) {
        elements[i] = Math.sqrt(elements[i]);
      }
    }

    public double elementSum() {
      double sum = 0.0;
      for (int i = 0; i < elements.length; i++) {
        sum += elements[i];
      }
      return sum;
    }

    public double get(int i) {
      return elements[i];
    }

    public void set(int i, double value) {
      elements[i] = value;
    }

    public void set(Vector v) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] = v.get(i);
      }
    }

  }

  /**
   * A class containing a single record. Each record contains the Quora answer
   * ID and the predictor vector. If the record is a training example then the
   * target classification is also included.
   */
  private class Record {

    private final Vector predictor;
    private final double target;
    private final String answerId;

    /**
     * Constructor for test records
     */
    public Record(Vector predictor, String answerId) {
      this.predictor = predictor;
      this.answerId = answerId;
      this.target = 0;
    }

    /**
     * Constructor for training records
     */
    public Record(Vector predictor, String answerId, double target) {
      this.predictor = predictor;
      this.answerId = answerId;
      this.target = target;
    }

    public Vector getPredictor() {
      return predictor;
    }

    public double getTarget() {
      return target;
    }

    public String getAnswerId() {
      return answerId;
    }

  }

  /**
   * Class for storing a list of records. The constructor reads the records from
   * the training or test sections of the input file as appropriate
   */
  private class RecordList extends ArrayList<Record> {

    private static final long serialVersionUID = 1L;
    private final int numFeatures; // number of features in predictor

    /**
     * Constructor which loads records from training part of input file
     *
     * @param sc
     *          scanner to read training data from
     */
    public RecordList(Scanner sc) {
      String[] dimString = sc.nextLine().split("\\s");
      int numRecords = Integer.parseInt(dimString[0]);
      numFeatures = Integer.parseInt(dimString[1]);
      for (int i = 0; i < numRecords; i++) {
        Vector predictor = new Vector(numFeatures);
        String[] recordString = sc.nextLine().split("\\s");
        // read Quora answer ID
        String answerId = recordString[0];
        // read training targets and convert from -1/+1 to 0/1
        double target = (Double.parseDouble(recordString[1]) + 1) / 2;
        // read predictor vector
        for (int fIndex = 0; fIndex < numFeatures; fIndex++) {
          String featureString = recordString[2 + fIndex];
          predictor.set(fIndex, Double.parseDouble(featureString
              .substring(featureString.indexOf(":") + 1)));
        }
        add(new Record(predictor, answerId, target));
      }
    }

    /**
     * Constructor which loads records from test part of input file
     *
     * @param sc
     *          scanner from which test data will be read
     */
    public RecordList(Scanner sc, int numFeatures) {
      this.numFeatures = numFeatures;
      String[] dimString = sc.nextLine().split("\\s");
      int numRecords = Integer.parseInt(dimString[0]);
      for (int i = 0; i < numRecords; i++) {
        Vector predictor = new Vector(numFeatures);
        String[] recordString = sc.nextLine().split("\\s");
        // read Quora answer ID
        String answerId = recordString[0];
        // read predictor vector
        for (int fIndex = 0; fIndex < numFeatures; fIndex++) {
          String featureString = recordString[1 + fIndex];
          predictor.set(fIndex, Double.parseDouble(featureString
              .substring(featureString.indexOf(":") + 1)));
        }
        add(new Record(predictor, answerId));
      }
    }

    public int getNumFeatures() {
      return numFeatures;
    }

  }

  /**
   * Class which supports normalization of records
   */
  private class Normalizer {

    private final Vector mean;
    private final Vector varSqrt;

    /**
     * Constructor normalizes supplied recordList and calculates elementwise
     * mean and variance
     */
    public Normalizer(RecordList recordList) {

      // calculate mean of each feature
      mean = new Vector(recordList.getNumFeatures());
      for (Record record : recordList) {
        mean.add(record.getPredictor());
      }
      mean.divide(recordList.size());

      // normalize recordList by feature means
      for (Record record : recordList) {
        record.getPredictor().subtract(mean);
      }

      // calculate variance of each feature
      varSqrt = new Vector(recordList.getNumFeatures());
      Vector predictorSquared = new Vector(recordList.getNumFeatures());
      for (Record record : recordList) {
        predictorSquared.set(record.getPredictor());
        predictorSquared.elementwiseSquare();
        varSqrt.add(predictorSquared);
      }
      varSqrt.divide(recordList.size());
      varSqrt.sqrt();

      // normalize recordList by feature variances
      for (Record record : recordList) {
        record.getPredictor().divide(varSqrt);
      }

    }

    /**
     * Normalize a Vector using the mean and variance calculated from supplied
     * recordList during instantiation
     *
     * @param vector
     *          Vector to be normalized
     */
    public void normalize(Vector vector) {
      vector.subtract(mean);
      vector.divide(varSqrt);
    }

  }

  /**
   * Class for training of a classifier
   */
  private class Trainer {

    // training rate for stochastic gradient descent
    private final double trainingRate;

    public Trainer(double trainingRate) {
      this.trainingRate = trainingRate;
    }

    /**
     * Train logistic regression coefficients using provided test records
     */
    public void train(Classifier classifier, RecordList testRecords) {

      for (Record record : testRecords) {
        // classify test record
        double estimate = classifier.classify(record.getPredictor());
        // determine classification error
        double estError = estimate - record.getTarget();
        // update regression coefficients using SGD
        Vector deltaTheta = record.getPredictor().clone();
        deltaTheta.multiply(-estError * trainingRate / testRecords.size());
        classifier.getRegressionCoeffs().add(deltaTheta);
      }

    }

  }

  /**
   * A class which implements classification using logistic regression
   */
  private class Classifier {

    // regression coefficients, trained using Trainer
    private final Vector regressionCoeffs;
    // normalizer used to normalize predictor vectors
    private final Normalizer normalizer;
    // number of features in predictor vector
    private final int numFeatures;

    /**
     * Instantiates a classifier with a particular training set. A normalizer
     * will be instantiated, storing the mean and variance of the training set
     * for future use during classification
     */
    public Classifier(RecordList trainingSet) {
      normalizer = new Normalizer(trainingSet);
      numFeatures = trainingSet.getNumFeatures();
      regressionCoeffs = new Vector(numFeatures);
    }

    /**
     * Classify a predictor vector
     *
     * @param predictor
     *          predictor vector to be classified
     * @return soft classification
     */
    public double classify(Vector predictor) {
      Vector normalizedPredictor = predictor.clone();
      normalizer.normalize(normalizedPredictor);
      normalizedPredictor.multiply(regressionCoeffs);
      return sigmoid(normalizedPredictor.elementSum());
    }

    /**
     * Sigmoid function
     *
     * @param z
     *          a double
     * @return double containing 1/(1 + exp(-z))
     */
    private double sigmoid(double z) {
      return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Return regression coefficients
     *
     * @return regression coefficient vector
     */
    public Vector getRegressionCoeffs() {
      return regressionCoeffs;
    }

    /**
     * Run classification for a set of queries and output the result to
     * System.out
     *
     * @param sc
     *          Scanner for reading queries
     */
    public void classifyQueries(Scanner sc) {
      RecordList classificationSet = new RecordList(sc, numFeatures);
      for (Record record : classificationSet) {
        // classify
        double classification = classify(record.getPredictor());
        // output result to System.out
        if (classification > 0.5) {
          System.out.printf("%s +1\n", record.getAnswerId());
        } else {
          System.out.printf("%s -1\n", record.getAnswerId());
        }
      }
    }

  }

  /**
   * Run classification using input data from scanner sc
   *
   * @param sc
   *          Scanner for reading training set and queries
   */
  public void run(Scanner sc) {
    double trainingRate = 5.0;
    RecordList trainingSet = new RecordList(sc);
    Classifier classifier = new Classifier(trainingSet);
    Trainer trainer = new Trainer(trainingRate);
    trainer.train(classifier, trainingSet);
    classifier.classifyQueries(sc);
  }

  /**
   * Run classification. If no arguments are supplied input is read from
   * standard input and output is written to standard output If an argument is
   * supplied then input is read from the file specified by the argument
   */
  public static void main(String[] args) throws FileNotFoundException {
    Scanner sc; // input file
    if (args.length > 0) { // input stream from file (for testing)
      BufferedReader in = new BufferedReader(new FileReader(new File(args[0])));
      sc = new Scanner(in);
    } else { // input streamed from System.in (used in competition)
      sc = new Scanner(System.in);
    }
    // run classification
    QuoraClassifier classifier = new QuoraClassifier();
    classifier.run(sc);
  }

}
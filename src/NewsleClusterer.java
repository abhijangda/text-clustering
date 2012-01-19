import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Solution for Newsle Clustering question from CodeSprint 2012. This class
 * implements clustering of text documents using Cosine or Jaccard distance
 * between the feature vectors of the documents together with k means
 * clustering. The number of clusters is adapted so that the ratio of the
 * intracluster to intercluster distance is below a specified threshold.
 */
@SuppressWarnings("serial")
public class NewsleClusterer {

  /**
   * Vector helper class. Supports basic vector operations (add, multiply,
   * divide)
   */
  private class Vector {

    double[] elements;

    public Vector(int vectorSize) {
      elements = new double[vectorSize];
    }

    public Vector clone() {
      Vector vectorClone = new Vector(elements.length);
      for (int i = 0; i < elements.length; i++) {
        vectorClone.set(i, elements[i]);
      }
      return vectorClone;
    }

    public void increment(int i) {
      elements[i]++;
    }

    public double get(int i) {
      return elements[i];
    }

    public void set(int i, double value) {
      elements[i] = value;
    }

    public void add(Vector vector) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] += vector.get(i);
      }
    }

    public void multiply(double multiplier) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] *= multiplier;
      }
    }

    /**
     * Elementwise multiplication
     */
    public void multiply(Vector vector) {
      for (int i = 0; i < elements.length; i++) {
        double multiplier = vector.get(i);
        if (elements[i] == 0.0 || multiplier == 0.0) {
          elements[i] = 0.0;
        } else {
          elements[i] *= multiplier;
        }
      }
    }

    public void divide(double divisor) {
      for (int i = 0; i < elements.length; i++) {
        elements[i] /= divisor;
      }
    }

    /**
     * Elementwise inversion
     */
    public void invert() {
      for (int i = 0; i < elements.length; i++) {
        elements[i] = 1.0 / elements[i];
      }
    }

    /**
     * Elementwise log operation
     */
    public void log() {
      for (int i = 0; i < elements.length; i++) {
        elements[i] = Math.log(elements[i]);
      }
    }

    /**
     * Return maximal element
     *
     * @return double containing the maximum element of the vector
     */
    public double max() {
      double maxValue = Double.MIN_VALUE;
      for (int i = 0; i < elements.length; i++) {
        maxValue = Math.max(maxValue, elements[i]);
      }
      return maxValue;
    }

    /**
     * L2 norm of vector
     */
    private double norm() {
      double normSquared = 0.0;
      for (int i = 0; i < elements.length; i++) {
        normSquared += elements[i] * elements[i];
      }
      return Math.sqrt(normSquared);
    }

    /**
     * Inner product of two vectors
     */
    public double innerProduct(Vector vector2) {
      double innerProduct = 0;
      for (int i = 0; i < elements.length; i++) {
        innerProduct += elements[i] * vector2.get(i);
      }
      return innerProduct;
    }

  }

  /**
   * Class containing an individual document
   */
  private class Document implements Comparable<Document> {

    @SuppressWarnings("unused")
    // document title, contents and ID
    private final String title;
    private final String contents;
    private final long id;
    // whether document has been allocated to a cluster
    private boolean allocated;
    // document word histogram
    private Vector histogram;
    // encoded document vector (TF-IDF)
    private Vector vector;
    // precalculated document vector norms
    private double norm;

    public Document(long id, String title, String contents) {
      this.id = id;
      this.title = title;
      this.contents = contents;
    }

    public void setVector(Vector vector) {
      this.vector = vector;
    }

    public boolean isAllocated() {
      return allocated;
    }

    /**
     * Mark document as having been allocated to a cluster
     */
    public void setIsAllocated() {
      allocated = true;
    }

    /**
     * Mark document as not being allocated to a cluster
     */
    public void clearIsAllocated() {
      allocated = false;
    }

    public long getId() {
      return id;
    }

    public String getContents() {
      return contents;
    }

    public Vector getVector() {
      return vector;
    }

    public Vector getHistogram() {
      return histogram;
    }

    public void setHistogram(Vector histogram) {
      this.histogram = histogram;
    }

    public void setNorm(double norm) {
      this.norm = norm;
    }

    public double getNorm() {
      return norm;
    }

    /**
     * Allow documents to be sorted by ID
     */
    public int compareTo(Document document) {
      if (id > document.getId()) {
        return 1;
      } else if (id < document.getId()) {
        return -1;
      } else {
        return 0;
      }
    }

  }

  /**
   * Class storing a collection of documents to be clustered
   */
  private class DocumentList extends ArrayList<Document> {

    /**
     * Parse input string into document ID, document title and contents then
     * encode into feature vector using encoder
     *
     * @param input
     *          String containing document fields
     * @param encoder
     *          Encoder to use for encoding document into feature vector
     */
    public DocumentList(String input) {
      StringTokenizer st = new StringTokenizer(input, "{");
      int numDocuments = st.countTokens() - 1;
      String record = st.nextToken(); // skip empty split to left of {
      Pattern pattern =
          Pattern
              .compile("\"content\": \"(.*)\", \"id\": (.*), \"title\": \"(.*)\"");
      for (int i = 0; i < numDocuments; i++) {
        record = st.nextToken();
        Matcher matcher = pattern.matcher(record);
        if (matcher.find()) {
          long documentID = Long.parseLong(matcher.group(2));
          String title = matcher.group(3);
          String contents = matcher.group(1);
          add(new Document(documentID, title, contents));
        }
      }
    }

    /**
     * Mark all documents as not being allocated to a cluster
     */
    public void clearIsAllocated() {
      for (Document document : this) {
        document.clearIsAllocated();
      }
    }

  }

  /**
   * Interface for encoders which encode documents into feature vectors
   */
  private interface Encoder {

    public void encode(DocumentList documentList);

  }

  /**
   * Implementation of Encoder which uses Term Frequency - Inverse Document
   * Frequency (TF-IDF) encoding
   */
  private class TfIdfEncoder implements Encoder {

    // number of features to be used in feature vector
    private final int numFeatures;
    // inverse document frequency used for normalization of feature vectors
    private Vector inverseDocumentFrequency = null;

    public TfIdfEncoder(int numFeatures) {
      this.numFeatures = numFeatures;
    }

    /**
     * Calculate word histogram for document and store in histogram field
     */
    private void calcHistogram(Document document) {

      // Calculate word histogram for document
      String[] words = document.getContents().split("[^\\w]+");
      Vector histogram = new Vector(numFeatures);
      for (int i = 0; i < words.length; i++) {
        int hashCode = hashWord(words[i]);
        histogram.increment(hashCode);
      }
      document.setHistogram(histogram);
    }

    /**
     * Calculate word histogram for all documents in a DocumentList
     */
    private void calcHistogram(DocumentList documentList) {
      for (Document document : documentList) {
        calcHistogram(document);
      }
    }

    /**
     * Calculate inverse document frequency for DocumentList. Assumes word
     * histograms for constituent documents have already been calculated
     */
    private void calcInverseDocumentFrequency(DocumentList documentList) {

      inverseDocumentFrequency = new Vector(numFeatures);
      for (Document document : documentList) {
        for (int i = 0; i < numFeatures; i++) {
          if (document.getHistogram().get(i) > 0) {
            inverseDocumentFrequency.increment(i);
          }
        }
      }
      inverseDocumentFrequency.invert();
      inverseDocumentFrequency.multiply(documentList.size());
      inverseDocumentFrequency.log();
    }

    /**
     * Encode document using Term Frequency - Inverse Document Frequency
     */
    private void encode(Document document) {

      // Normalize word histogram by maximum word frequency
      Vector vector = document.getHistogram().clone();
      // Allow histogram to be deallocated as it is no longer needed
      document.setHistogram(null);
      vector.divide(vector.max());
      // Normalize by inverseDocumentFrequency
      vector.multiply(inverseDocumentFrequency);
      // Store feature vecotr in document
      document.setVector(vector);
      // Precalculate norm for use in distance calculations
      document.setNorm(vector.norm());

    }

    /**
     * Hash word into integer between 0 and numFeatures-1. Used to form document
     * feature vector
     *
     * @param word
     *          String to be hashed
     * @return hashed integer
     */
    private int hashWord(String word) {
      return Math.abs(word.hashCode()) % numFeatures;
    }

    /**
     * Encode all documents within a DocumentList
     */
    public void encode(DocumentList documentList) {

      calcHistogram(documentList);
      calcInverseDocumentFrequency(documentList);
      for (Document document : documentList) {
        encode(document);
      }

    }

  }

  /**
   * Class representing a cluster of Documents
   */
  private class Cluster extends ArrayList<Document> implements
      Comparable<Cluster> {

    // cluster centroid
    private Vector centroid;
    // norm of cluster centroid
    private double centroidNorm;

    /**
     * Instantiate a cluster with a single member document
     */
    public Cluster(Document document) {
      super();
      add(document);
      centroid = document.getVector().clone();
      centroidNorm = centroid.norm();
    }

    /**
     * Allows sorting of Clusters by comparing ID of first document
     */
    public int compareTo(Cluster cluster) {
      return get(0).compareTo(cluster.get(0));
    }

    /**
     * Add document to cluster and mark document as allocated
     */
    public boolean add(Document document) {
      super.add(document);
      document.setIsAllocated();
      return true;
    }

    /**
     * Update centroids and centroidNorms for a specific cluster
     *
     * @param clusterIndex
     *          cluster to update
     */
    private void updateCentroid() {
      centroid = null;
      for (Document document : this) {
        if (centroid == null) {
          centroid = document.getVector().clone();
        } else {
          centroid.add(document.getVector());
        }
      }
      centroid.divide(size());
      centroidNorm = centroid.norm();
    }

    public Vector getCentroid() {
      return centroid;
    }

    public double getCentroidNorm() {
      return centroidNorm;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("[");
      for (int i = 0; i < this.size(); i++) {
        sb.append(this.get(i).getId());
        if (i < this.size() - 1) {
          sb.append(", ");
        }
      }
      sb.append("]");
      return sb.toString();
    }

  }

  /**
   * Class representing a list of clusters. This is the output of the clustering
   * process
   */

  private class ClusterList extends ArrayList<Cluster> {

    /**
     * Instantiate a list of clusters with a given initial capacity
     *
     * @param initialCapacity
     */
    public ClusterList(int initialCapacity) {
      super(initialCapacity);
    }

    /**
     * Update centroids of all clusters within ClusterList
     */
    public void updateCentroids() {
      for (Cluster cluster : this) {
        cluster.updateCentroid();
      }
    }

    /**
     * Find document with maximum distance to clusters in ClusterList. Distance
     * to ClusterList is defined as the minimum of the distances to each
     * constituent Cluster's centroid. This method is used during the cluster
     * initialization in k means clustering
     *
     * @param distance
     *          distance measure to use
     * @param documentList
     *          list of documents to search
     * @return furthest document
     */
    public Document findFurthestDocument(Distance distance,
        DocumentList documentList) {
      double furthestDistance = Double.MIN_VALUE;
      Document furthestDocument = null;
      for (Document document : documentList) {
        if (!document.isAllocated()) {
          double documentDistance = distance.calcDistance(document, this);
          if (documentDistance > furthestDistance) {
            furthestDistance = documentDistance;
            furthestDocument = document;
          }
        }
      }
      return furthestDocument;
    }

    /**
     * Clear out documents from within each cluster. Used to cleanup at the end
     * of each iteration of k means
     */
    private void emptyClusters() {
      for (Cluster cluster : this) {
        cluster.clear();
      }
    }

    /**
     * Find cluster whose centroid is closest to a document
     *
     * @param distance
     *          distance measure to use
     * @param document
     * @return Cluster closest to document
     */
    private Cluster findNearestCluster(Distance distance, Document document) {
      Cluster nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      for (Cluster cluster : this) {
        double clusterDistance = distance.calcDistance(document, cluster);
        if (clusterDistance < nearestDistance) {
          nearestDistance = clusterDistance;
          nearestCluster = cluster;
        }
      }
      return nearestCluster;
    }

    /**
     * Calculate ratio of average intracluster distance to average intercluster
     * distance. Used to optimize number of clusters k
     *
     * @return ratio of average intracluster distance to average intercluster
     *         distance
     */
    private double calcIntraInterDistanceRatio(Distance distance) {
      if (this.size() > 1) {
        double interDist = calcInterClusterDistance(distance);
        if (interDist > 0.0) {
          return calcIntraClusterDistance(distance) / interDist;
        } else {
          return Double.MAX_VALUE;
        }
      } else {
        return Double.MAX_VALUE;
      }
    }

    /**
     * Calculate average intracluster distance, which is the average distance
     * between the constituent documents in a cluster and the cluster centroid
     *
     * @param distance
     *          distance measure to use
     */
    private double calcIntraClusterDistance(Distance distance) {
      double avgIntraDist = 0.0;
      int numDocuments = 0;
      for (Cluster cluster : this) {
        double clusterIntraDist = 0.0;
        for (Document document : cluster) {
          clusterIntraDist += distance.calcDistance(document, cluster);
        }
        numDocuments += cluster.size();
        avgIntraDist += clusterIntraDist;
      }
      return avgIntraDist / numDocuments;
    }

    /**
     * Calculate average intercluster distance, which is the distance between
     * cluster centroids
     *
     * @param distance
     *          distance measure to use
     */
    private double calcInterClusterDistance(Distance distance) {

      if (this.size() > 1) {
        double avgInterDist = 0.0;
        for (Cluster cluster1 : this) {
          for (Cluster cluster2 : this) {
            if (cluster1 != cluster2) {
              avgInterDist += distance.calcDistance(cluster1, cluster2);
            }
          }
        }
        // there are N*N-1 unique pairs of clusters
        avgInterDist /= (this.size() * (this.size() - 1));
        return avgInterDist;
      } else {
        return 0.0;
      }
    }

    /**
     * Sort order of documents within each cluster, then sort order of clusters
     * within ClusterList
     */
    public void sort() {
      for (Cluster cluster : this) {
        Collections.sort(cluster);
      }
      Collections.sort(this);
    }

    /**
     * Display clusters in sorted order
     */
    public String toString() {
      sort();
      StringBuilder sb = new StringBuilder();
      sb.append("[");
      for (int i = 0; i < this.size(); i++) {
        sb.append(this.get(i));
        if (i < this.size() - 1) {
          sb.append(", ");
        }
      }
      sb.append("]");
      return sb.toString();
    }

  }

  /**
   * Class for measuring the distance between documents, clusters and
   * clusterLists
   */
  private abstract class Distance {

    public double calcDistance(Document document, Cluster cluster) {

      return calcDistance(document.getVector(), cluster.getCentroid(),
          document.getNorm(), cluster.getCentroidNorm());

    }

    /**
     * Calculate minimum of the distances between a document and the centroids
     * of the clusters within a clusterList
     */
    public double calcDistance(Document document, ClusterList clusterList) {
      double distance = Double.MAX_VALUE;
      for (Cluster cluster : clusterList) {
        distance = Math.min(distance, calcDistance(document, cluster));
      }
      return distance;
    }

    public double calcDistance(Cluster cluster1, Cluster cluster2) {
      return calcDistance(cluster1.getCentroid(), cluster2.getCentroid(),
          cluster1.getCentroidNorm(), cluster2.getCentroidNorm());
    }

    public abstract double calcDistance(Vector vector1, Vector vector2,
        double norm1, double norm2);

  }

  /**
   * Class for calculating cosine distance between two vectors
   */
  private class CosineDistance extends Distance {

    public double calcDistance(Vector vector1, Vector vector2, double norm1,
        double norm2) {
      return 1.0 - vector1.innerProduct(vector2) / norm1 / norm2;
    }

  }

  /**
   * Class for caculating Jaccard distance between vectors
   */
  @SuppressWarnings("unused")
  private class JaccardDistance extends Distance {

    public double calcDistance(Vector vector1, Vector vector2, double norm1,
        double norm2) {
      double innerProduct = vector1.innerProduct(vector2);
      return Math.abs(1.0 - innerProduct / (norm1 + norm2 - innerProduct));
    }

  }

  /**
   * Class for a clusterer which clusters the documents within a DocumentList
   * into a ClusterList
   */
  private abstract class Clusterer {

    // distance measure to use when evaluating distance between documents
    Distance distance;

    public abstract ClusterList cluster(DocumentList documentList);

  }

  /**
   * An k means clustering implementation of the Clusterer class
   */
  private class KMeansClusterer extends Clusterer {

    // threshold used to determine number of clusters k
    private final double clusteringThreshold;
    // number of iterations to use in k means clustering
    private final int clusteringIterations;

    public KMeansClusterer(Distance distance, double clusteringThreshold,
        int clusteringIterations) {
      this.distance = distance;
      this.clusteringThreshold = clusteringThreshold;
      this.clusteringIterations = clusteringIterations;
    }

    /**
     * Run k means clustering on documentList. Number of clusters k is set to
     * the lowest value that ensures the intracluster to intercluster distance
     * ratio is above clusteringThreshold
     *
     * @return ClusterList containing the clusters
     */
    public ClusterList cluster(DocumentList documentList) {
      ClusterList clusterList = null;
      for (int k = 1; k <= documentList.size(); k++) {
        clusterList = runKMeansClustering(documentList, k);
        if (clusterList.calcIntraInterDistanceRatio(distance) < clusteringThreshold) {
          break;
        }
      }
      return clusterList;
    }

    /**
     * Run k means clustering on documentList for a fixed number of clusters k
     *
     * @param documentList
     *          documents to be clustered
     * @param k
     *          target number of clusters
     * @return ClusterList containing the clusters
     */
    private ClusterList runKMeansClustering(DocumentList documentList, int k) {
      ClusterList clusterList = new ClusterList(k);
      documentList.clearIsAllocated();
      // create 1st cluster using a random document
      Random rnd = new Random();
      int rndDocIndex = rnd.nextInt(k);
      Cluster initialCluster = new Cluster(documentList.get(rndDocIndex));
      clusterList.add(initialCluster);
      // create k-1 more clusters
      while (clusterList.size() < k) {
        // create new cluster containing furthest doc from existing clusters
        Document furthestDocument =
            clusterList.findFurthestDocument(distance, documentList);
        Cluster nextCluster = new Cluster(furthestDocument);
        clusterList.add(nextCluster);
      }

      // add remaining documents to one of the k existing clusters
      for (int iter = 0; iter < clusteringIterations; iter++) {
        for (Document document : documentList) {
          if (!document.isAllocated()) {
            Cluster nearestCluster =
                clusterList.findNearestCluster(distance, document);
            nearestCluster.add(document);
          }
        }
        // update centroids and centroidNorms
        clusterList.updateCentroids();
        // prepare for reallocation in next iteration
        if (iter < clusteringIterations - 1) {
          documentList.clearIsAllocated();
          clusterList.emptyClusters();
        }
      }
      return clusterList;
    }

  }

  /**
   * Run clustering on input string. Clustering process consists of parsing and
   * encoding documents, and then using Clusterer with a specific Distance
   * measure.
   *
   * @param input
   *          String containing documents
   */
  public void run(String input) {
    DocumentList documentList = new DocumentList(input);
    Encoder encoder = new TfIdfEncoder(10000);
    encoder.encode(documentList);
    Distance distance = new CosineDistance();
    Clusterer clusterer = new KMeansClusterer(distance, 0.4, 3);
    ClusterList clusterList = clusterer.cluster(documentList);
    System.out.println(clusterList.toString());
  }

  /**
   * Run clustering from command line. By default input is read from standard
   * input and output is written to standard output. If an optional argument is
   * used then input is read from the file specified
   */
  public static void main(String[] args) throws IOException {
    String input;
    if (args.length > 0) {
      BufferedReader in = new BufferedReader(new FileReader(new File(args[0])));
      input = in.readLine();
    } else {
      Scanner sc = new Scanner(System.in);
      input = sc.nextLine();
    }
    NewsleClusterer newsleClusterer = new NewsleClusterer();
    newsleClusterer.run(input);
  }

}

package com.cendrillon.clustering;

import java.util.Random;

/**
 * A class for clustering documents. Uses k-means clustering.
 */
public class Clusterer {
	private final double clusteringThreshold;
	private final int clusteringIterations;
	private final DistanceMetric distance;

	/**
	 * Construct a clusterer.
	 * 
	 * @param distance the distance metric to use for clustering
	 * @param clusteringThreshold the threshold used to determine the number of clusters k
	 * @param clusteringIterations the number of iterations to use in k-means clustering
	 */
	public Clusterer(DistanceMetric distance, double clusteringThreshold, int clusteringIterations) {
		this.distance = distance;
		this.clusteringThreshold = clusteringThreshold;
		this.clusteringIterations = clusteringIterations;
	}

	/**
	 * Run k-means clustering on the provided documentList. Number of clusters k is set to the lowest
	 * value that ensures the intracluster to intercluster distance ratio is above
	 * clusteringThreshold.
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

	/** Run k means clustering on documentList for a fixed number of clusters k. */
	private ClusterList runKMeansClustering(DocumentList documentList, int k) {
		ClusterList clusterList = new ClusterList();
		documentList.clearIsAllocated();
		// create 1st cluster using a random document
		Random rnd = new Random();
		int rndDocIndex = rnd.nextInt(k);
		Cluster initialCluster = new Cluster(documentList.get(rndDocIndex));
		clusterList.add(initialCluster);
		// create k-1 more clusters
		while (clusterList.size() < k) {
			// create new cluster containing furthest doc from existing clusters
			Document furthestDocument = clusterList.findFurthestDocument(distance, documentList);
			Cluster nextCluster = new Cluster(furthestDocument);
			clusterList.add(nextCluster);
		}

		// add remaining documents to one of the k existing clusters
		for (int iter = 0; iter < clusteringIterations; iter++) {
			for (Document document : documentList) {
				if (!document.isAllocated()) {
					Cluster nearestCluster = clusterList.findNearestCluster(distance, document);
					nearestCluster.add(document);
				}
			}
			// update centroids and centroidNorms
			clusterList.updateCentroids();
			// prepare for reallocation in next iteration
			if (iter < clusteringIterations - 1) {
				documentList.clearIsAllocated();
				clusterList.clear();
			}
		}
		return clusterList;
	}
}

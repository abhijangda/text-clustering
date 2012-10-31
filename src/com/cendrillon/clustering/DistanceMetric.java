package com.cendrillon.clustering;

/**
 * Class for measuring the distance between documents, clusters and clusterLists.
 */
public abstract class DistanceMetric {
	public double calcDistance(Cluster cluster1, Cluster cluster2) {
		return calcDistance(cluster1.getCentroid(), cluster2.getCentroid(), cluster1.getCentroidNorm(),
		    cluster2.getCentroidNorm());
	}

	public double calcDistance(Document document, Cluster cluster) {
		return calcDistance(document.getVector(), cluster.getCentroid(), document.getNorm(),
		    cluster.getCentroidNorm());
	}

	/**
	 * Calculate minimum of the distances between a document and the centroids of the clusters within
	 * a clusterList.
	 */
	public double calcDistance(Document document, ClusterList clusterList) {
		double distance = Double.MAX_VALUE;
		for (Cluster cluster : clusterList) {
			distance = Math.min(distance, calcDistance(document, cluster));
		}
		return distance;
	}

	/**
	 * Calculate distance between two Vectors. The precalculated Vector norms are passed in to speed
	 * up computation.
	 */
	protected abstract double
	    calcDistance(Vector vector1, Vector vector2, double norm1, double norm2);
}

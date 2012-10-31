package com.cendrillon.clustering;

import java.util.ArrayList;
import java.util.List;

/** Class representing a cluster of Documents. */
public class Cluster implements Comparable<Cluster> {
	private Vector centroid; // cluster centroid
	private double centroidNorm; // norm of cluster centroid
	private final ArrayList<Document> documents = new ArrayList<Document>();

	/** Construct a cluster with a single member document. */
	public Cluster(Document document) {
		documents.add(document);
		centroid = new Vector(document.getVector());
		centroidNorm = centroid.norm();
	}

	/**
	 * Add document to cluster and mark document as allocated
	 */
	public void add(Document document) {
		documents.add(document);
		document.setIsAllocated();
	}

	/**
	 * Remove all documents from a cluster.
	 */
	public void clear() {
		documents.clear();
	}

	/**
	 * Allows sorting of Clusters by comparing ID of first document.
	 */
	@Override
	public int compareTo(Cluster cluster) {
		return documents.get(0).compareTo(cluster.documents.get(0));
	}

	public Vector getCentroid() {
		return centroid;
	}

	public double getCentroidNorm() {
		return centroidNorm;
	}

	public List<Document> getDocuments() {
		return documents;
	}

	public int size() {
		return documents.size();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("[ ");
		for (Document document : documents) {
			sb.append(document.getId());
			sb.append(" ");
		}
		sb.append("]");
		return sb.toString();
	}

	/** Update centroids and centroidNorms for this cluster. */
	public void updateCentroid() {
		centroid = null;
		for (Document document : documents) {
			if (centroid == null) {
				centroid = new Vector(document.getVector());
			} else {
				centroid = centroid.add(document.getVector());
			}
		}
		centroid = centroid.divide(size());
		centroidNorm = centroid.norm();
	}
}

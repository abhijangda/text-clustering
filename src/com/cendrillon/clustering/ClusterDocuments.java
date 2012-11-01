package com.cendrillon.clustering;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Solution for Newsle Clustering question from CodeSprint 2012. This class implements clustering of
 * text documents using Cosine or Jaccard distance between the feature vectors of the documents
 * together with k means clustering. The number of clusters is adapted so that the ratio of the
 * intracluster to intercluster distance is below a specified threshold.
 */
public class ClusterDocuments {
	private static final int CLUSTERING_ITERATIONS = 3;
	private static final double CLUSTERING_THRESHOLD = 0.3;
	private static final int NUM_FEATURES = 10000;

	/**
	 * Cluster the text documents in the provided file. The clustering process consists of parsing and
	 * encoding documents, and then using Clusterer with a specific Distance measure.
	 */
	public static void main(String[] args) throws IOException {
		if (args.length != 1) {
			System.out.println("Usage: ClusterDocuments <filename>\n");
			System.exit(1);
		}
		BufferedReader in = new BufferedReader(new FileReader(new File(args[0])));
		String input = in.readLine();
		in.close();
		DocumentList documentList = new DocumentList(input);
		Encoder encoder = new TfIdfEncoder(NUM_FEATURES);
		encoder.encode(documentList);
		DistanceMetric distance = new CosineDistance();
		Clusterer clusterer = new KMeansClusterer(distance, CLUSTERING_THRESHOLD, CLUSTERING_ITERATIONS);
		ClusterList clusterList = clusterer.cluster(documentList);
		System.out.println(clusterList);
	}
}
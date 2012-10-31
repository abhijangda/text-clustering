package com.cendrillon.clustering;

/**
 * Class containing an individual document.
 */
public class Document implements Comparable<Document> {
	private final String contents;
	private final long id;
	private boolean allocated;
	private Vector histogram;
	private Vector vector;
	private double norm;

	public Document(long id, String contents) {
		this.id = id;
		this.contents = contents;
	}

	/** Mark document as not being allocated to a cluster. */
	public void clearIsAllocated() {
		allocated = false;
	}

	/** Allow documents to be sorted by ID. */
	@Override
	public int compareTo(Document document) {
		if (id > document.getId()) {
			return 1;
		} else if (id < document.getId()) {
			return -1;
		} else {
			return 0;
		}
	}

	public String getContents() {
		return contents;
	}

	/** Get document word histogram. */
	public Vector getHistogram() {
		return histogram;
	}

	public long getId() {
		return id;
	}

	/** Get norm of document feature vector. */
	public double getNorm() {
		return norm;
	}

	/** Get encoded document feature vector. */
	public Vector getVector() {
		return vector;
	}

	/** Determine whether document has been allocated to a cluster. */
	public boolean isAllocated() {
		return allocated;
	}

	public void setHistogram(Vector histogram) {
		this.histogram = histogram;
	}

	/** Mark document as having been allocated to a cluster. */
	public void setIsAllocated() {
		allocated = true;
	}

	public void setNorm(double norm) {
		this.norm = norm;
	}

	public void setVector(Vector vector) {
		this.vector = vector;
	}
}

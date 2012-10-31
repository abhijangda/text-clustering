package com.cendrillon.clustering;

import java.util.Arrays;

/**
 * Vector helper class. Supports basic vector operations (add, multiply, divide etc.).
 */
public class Vector {
	private final double[] elements;

	public Vector(double[] original) {
		elements = Arrays.copyOf(original, original.length);
	}

	public Vector(int size) {
		elements = new double[size];
	}

	public Vector(Vector vector) {
		elements = Arrays.copyOf(vector.elements, vector.elements.length);
	}

	/** Add the provided Vector to this. */
	public void add(Vector operand) {
		for (int i = 0; i < elements.length; i++) {
			set(i, get(i) + operand.get(i));
		}
	}

	/** Divide this by the provided divisor. */
	public void divide(double divisor) {
		for (int i = 0; i < elements.length; i++) {
			set(i, get(i) / divisor);
		}
	}

	/** Get the element of this Vector at the specified index. */
	public double get(int i) {
		return elements[i];
	}

	/** Apply elementwise increment to this. */
	public void increment(int i) {
		set(i, get(i) + 1);
	}

	/**
	 * Calculate the inner product of this with the provided vector.
	 */
	public double innerProduct(Vector vector) {
		double innerProduct = 0;
		for (int i = 0; i < elements.length; i++) {
			innerProduct += get(i) * vector.get(i);
		}
		return innerProduct;
	}

	/** Apply elementwise inversion to this. */
	public void invert() {
		for (int i = 0; i < elements.length; i++) {
			set(i, 1 / get(i));
		}
	}

	/** Apply elementwise log to this. */
	public void log() {
		for (int i = 0; i < elements.length; i++) {
			set(i, Math.log(get(i)));
		}
	}

	/** Return maximal element. */
	public double max() {
		double maxValue = Double.MIN_VALUE;
		for (int i = 0; i < elements.length; i++) {
			maxValue = Math.max(maxValue, get(i));
		}
		return maxValue;
	}

	/** Multiply this with the provided scalar multiplier. */
	public void multiply(double multiplier) {
		for (int i = 0; i < elements.length; i++) {
			set(i, get(i) * multiplier);
		}
	}

	/** Multiply this with the provided vector multiplier. */
	public void multiply(Vector multiplier) {
		for (int i = 0; i < elements.length; i++) {
			if (get(i) == 0 || multiplier.get(i) == 0) {
				set(i, 0);
			} else {
				set(i, multiplier.get(i) * get(i));
			}
		}
	}

	/** Calculate the L2 norm of this. */
	public double norm() {
		double normSquared = 0.0;
		for (int i = 0; i < elements.length; i++) {
			normSquared += get(i) * get(i);
		}
		return Math.sqrt(normSquared);
	}

	/** Set the specified element of this. */
	public void set(int i, double value) {
		elements[i] = value;
	}
}
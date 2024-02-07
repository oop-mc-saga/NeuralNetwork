package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;
import java.util.function.DoubleFunction;

/**
 * Neuron class
 *
 * @author tadaki
 */
public class Neuron {

    private final String label;
    private final List<Double> weight;//the last elements is the threshold
    private final DoubleFunction<Double> function;//responce function
    private boolean fixOutput = false;//true corresponding to input of the threshold
    private final String nl = System.getProperty("line.separator");

    private double z;
    private double y;
    private List<Double> input;

    /**
     * Construstor
     *
     * @param label label of this neuron
     * @param weight weight list
     * @param function response function
     */
    public Neuron(String label, List<Double> weight,
            DoubleFunction<Double> function) {
        this.label = label;
        this.weight = weight;
        this.function = function;
    }

    /**
     *
     * @param label label of this neuron
     * @param function weight list
     * @param n the number of input, weights are randomly generated
     * @param random
     */
    public Neuron(String label, DoubleFunction<Double> function,
            int n, Random random) {
        weight = new ArrayList<>();
        for (int i = 0; i < n; i++) {//weights are randomly generated.
            double x = random.nextDouble() - 0.5;
            weight.add(x);
        }
        this.label = label;
        this.function = function;
    }

    /**
     * Response to input
     *
     * @param input
     * @return
     */
    public Double response(List<Double> input) {
        z = Neuron.product(input, this.weight);
        this.input = input;
        if (fixOutput) {
            y = 1.;
        } else {
            y = function.apply(z);
        }
        return y;
    }

    /**
     * Inner product of two vectors
     *
     * @param a
     * @param b
     * @return
     */
    static public double product(List<Double> a, List<Double> b) {
        if (a.size() != b.size()) {
            String message = "size " + a.size()
                    + " is not equal to size" + b.size();
            throw new IllegalArgumentException(message);
        }
        double x = 0.;
        for (int i = 0; i < a.size(); i++) {
            x += a.get(i) * b.get(i);
        }
        return x;
    }

    /**
     * return string expression of the output to the input
     *
     * @param input
     * @return
     */
    public String strResponse(List<Double> input) {
        Double r = response(input);
        StringJoiner sj = new StringJoiner(",", "(", ")");
        for (int i = 0; i < input.size() - 1; i++) {
            sj.add(String.valueOf(input.get(i)));
        }
        String str = label + sj.toString() + "=" + r;
        return str;
    }

    /**
     * Return all possible output
     *
     * @return
     */
    public String getAllResponseStr() {
        List<List<Double>> data = allInput(weight.size());
        StringBuilder sb = new StringBuilder();
        for (List<Double> inputData : data) {
            double r = response(inputData);
            sb.append(inputData).append(":");
            sb.append(String.format("%.2f", r));
            sb.append(nl);
        }
        return sb.toString();
    }

    public void normalizeWeight() {
        normalize(weight);
    }

    /**
     * Normalize vector
     *
     * @param vector
     */
    public static void normalize(List<Double> vector) {
        double a = product(vector, vector);
        a = Math.sqrt(a);
        for (int i = 0; i < vector.size(); i++) {
            double w = vector.get(i);
            vector.set(i, w / a);
        }
    }

    /**
     * Update weights responding to input
     *
     * @param input
     * @param coeff
     */
    public void update(List<Double> input, double coeff) {
        if (fixOutput) {
            return;
        }
        for (int i = 0; i < weight.size(); i++) {
            double w = weight.get(i) - coeff * input.get(i);
            weight.set(i, w);
        }
    }

    //********* setters and getters *****************
    public List<Double> getWeight() {
        List<Double> list = Collections.synchronizedList(new ArrayList<>());
        weight.stream().forEachOrdered(w -> list.add(w));
        return list;
    }

    public void setFixOutput(boolean fixOutput) {
        this.fixOutput = fixOutput;
    }

    public boolean isFixOutput() {
        return fixOutput;
    }

    public double getZ() {
        return z;
    }

    public double getY() {
        return y;
    }

    public List<Double> getInput() {
        return input;
    }

    /**
     * Generate all possible input patters
     *
     * @param n
     * @return
     */
    public static List<List<Double>> allInput(int n) {
        List<List<Double>> inputList
                = Collections.synchronizedList(new ArrayList<>());
        List<Double> list = new ArrayList<>();
        createInputList(0, list, inputList, n);
        return inputList;
    }

    private static void createInputList(int k, List<Double> list,
            List<List<Double>> inputList, int n) {
        if (k > n - 2) {
            list.add(1.);
            inputList.add(list);
            return;
        }
        List<Double> newList = new ArrayList<>(list);
        newList.add(1.);
        createInputList(k + 1, newList, inputList, n);
        newList = new ArrayList<>(list);
        newList.add(-1.);
        createInputList(k + 1, newList, inputList, n);
    }

}

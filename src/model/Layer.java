package model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Layer class
 *
 * @author tadaki
 */
public class Layer {

    private final List<Neuron> neurons;
    private final int numInput;

    /**
     * Constructor
     *
     * @param numNuerons the number of neurons
     * @param numInput the number of inputs
     * @param function response function
     * @param fixOutput the last neuron always emits 1 if set true
     * @param random
     */
    public Layer(int numNuerons, int numInput, DoubleFunction<Double> function,
            boolean fixOutput, Random random) {
        neurons = new ArrayList<>();
        this.numInput = numInput;
        for (int i = 0; i < numNuerons; i++) {
            neurons.add(new Neuron(String.valueOf(i), function,
                    numInput, random));
        }
        neurons.getLast().setFixOutput(fixOutput);
    }

    public Layer(int numInput, List<Neuron> neurons) {
        this.numInput = numInput;
        this.neurons = neurons;
    }

    /**
     * Response from this layer
     *
     * @param input
     * @return
     */
    public List<Double> response(List<Double> input) {
        if (input.size() != numInput) {
            String message = "input size " + input.size()
                    + " is not equal to asuumed value" + numInput;
            throw new IllegalArgumentException(message);
        }
        List<Double> output = new ArrayList<>();
        neurons.stream().forEachOrdered(
                n -> output.add(n.response(input)));
        return output;
    }

    public void normalizeWeight() {
        neurons.forEach(n -> n.normalizeWeight());
    }

    //********* setters and getters *****************
    public int getNumNuerons() {
        return neurons.size();
    }

    public int getNumInput() {
        return numInput;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public Neuron getNeuron(int i) {
        return neurons.get(i);
    }

    public void setFixOutput(boolean f) {
        neurons.getLast().setFixOutput(f);
    }

}

package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Multi-layer Neural Network
 * 
 * @author tadaki
 */
public class MultiLayer {

    private final List<Layer> layers;
    private final List<DoubleFunction<Double>> derivatives;
    private final boolean debug = false;

    /**
     * Constructor
     *
     * Ensure that data for each layer is provided in sequential order,
     * beginning with the input layer. However, the input layer is not included.
     *
     * @param numInput
     * @param numNeurons
     * @param functions
     * @param derivatives
     * @param random
     */
    public MultiLayer(int numInput, List<Integer> numNeurons,
            List<DoubleFunction<Double>> functions,
            List<DoubleFunction<Double>> derivatives, Random random) {
        this.derivatives = derivatives;
        int numberOfLyers = numNeurons.size();

        // Layers are instantiated starting with the input layer.
        layers = Collections.synchronizedList(new ArrayList<>());
        Layer input = new Layer(numNeurons.getFirst(), numInput,
                functions.getFirst(), true, random);
        layers.add(input);
        // Add layers from input side to output side
        for (int i = 1; i < numberOfLyers; i++) {
            Layer layer = new Layer(numNeurons.get(i),
                    layers.get(i - 1).getNumNuerons(),
                    functions.get(i), true, random);
            layers.add(layer);
        }
        layers.getLast().setFixOutput(false);
        layers.forEach(l -> l.normalizeWeight());
    }

    /**
     *
     * @param layers
     * @param derivatives
     */
    public MultiLayer(List<Layer> layers,
            List<DoubleFunction<Double>> derivatives) {
        this.derivatives = derivatives;
        this.layers = layers;
    }

    /**
     * Response to the input from the outside
     *
     * @param input the input from the outside
     * @return response of the system
     */
    public List<Double> response(List<Double> input) {
        List<Double> output = null;
        int n = layers.size();
        // Note: the order of layers starts from the response layer
        for (int i = 0; i < n; i++) {
            output = layers.get(i).response(input);
            input = output;
        }
        return output;
    }

    public Layer getLayer(int i) {
        if (i >= layers.size()) {
            return null;
        }
        return layers.get(i);
    }

    /**
     * Learn by back-propagation
     *
     * @param input input to the system
     * @param correctResponse correct responce
     * @param c learning coefficient
     */
    public void learn(List<Double> input, CorrectResponse correctResponse,
            double c) {
        List<Double> correctOutput = correctResponse.apply(input);
        List<Double> rList = new ArrayList<>();
        // Learning process starts from the response layer and propagates backward
        int n = layers.size();
        for (int layerIndex = n - 1; layerIndex >= 0; layerIndex--) {
            Layer layer = layers.get(layerIndex);
            if (rList.isEmpty()) {//response layer
                for (int i = 0; i < layer.getNumNuerons(); i++) {
                    Neuron neuron = layer.getNeuron(i);
                    // Descrepacy from the currect output
                    double r0 = neuron.getY() - correctOutput.get(i);
                    double r = r0 * derivatives.get(layerIndex).apply(neuron.getZ());
                    if (debug) {
                        StringBuilder sb = new StringBuilder();
                        sb.append(neuron.getY()).append(" ");
                        sb.append(correctOutput.get(i));
                        sb.append(" -> ").append(r0);
                        sb.append(" : ").append(r);
                        System.out.println(sb.toString());
                    }
                    rList.add(r);
                }
            } else {//layers except response layer 

                Layer nextLayer = layers.get(layerIndex + 1);// forward direction
                List<Double> rNewList = new ArrayList<>();
                for (int i = 0; i < layer.getNumNuerons(); i++) {
                    Neuron neuron = layer.getNeuron(i);
                    double r = 0.;
                    for (int k = 0; k < nextLayer.getNumNuerons(); k++) {
                        r += rList.get(k) * nextLayer.getNeuron(k).getWeight().get(i);
                    }
                    r *= derivatives.get(layerIndex).apply(neuron.getZ());
                    rNewList.add(r);
                }
                rList = rNewList;
            }
            //updating weight
            updateWeight(layer, rList, c);
        }
//        layers.forEach(l -> l.normalizeWeight());
    }

    private void updateWeight(Layer layer, List<Double> rList, double c) {
        for (int i = 0; i < layer.getNumNuerons(); i++) {
            Neuron neuron = layer.getNeuron(i);
            neuron.update(neuron.getInput(), c * rList.get(i));
        }
    }

    /**
     * Get error value
     *
     * @param input
     * @param correctResponse
     * @return
     */
    public double getError(List<Double> input, CorrectResponse correctResponse) {
        double error = 0.;
        Layer layer = layers.getLast();
        int n = layer.getNumNuerons();
        List<Double> correctOutput = correctResponse.apply(input);
        for (int i = 0; i < n; i++) {
            double v = (layer.getNeuron(i).getY() - correctOutput.get(i));
            error += v * v;
        }
        return error / 2. / n;
    }
}

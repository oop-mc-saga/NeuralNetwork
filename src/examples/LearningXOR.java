package examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleFunction;
import model.*;

/**
 *
 * @author tadaki
 */
public class LearningXOR {

    private final MultiLayer sys;

    public final CorrectResponse cr = input -> {
        double y = input.get(0) + input.get(1);
        double z = 1.;
        if (y > .5 || y < -.5) {
            z = -1.;
        }
        List<Double> output = new ArrayList<>();
        output.add(z);
        return output;
    };

    public LearningXOR(Random random) {
        //response function and its derivative
        double a = 1.;
        DoubleFunction<Double> f = x -> Math.tanh(a * x);
        DoubleFunction<Double> g = x -> a / Math.cosh(a * x) / Math.cosh(a * x);
        List<DoubleFunction<Double>> functions = new ArrayList<>();
        functions.add(f);
        functions.add(f);
        List<DoubleFunction<Double>> derivatives = new ArrayList<>();
        derivatives.add(g);
        derivatives.add(g);

        Integer[] nList = {3, 1};
        List<Integer> numNeurons = Arrays.asList(nList);
        sys = new MultiLayer(3, numNeurons, functions, derivatives, random);
    }

    public List<Double> response(List<Double> input) {
        return sys.response(input);
    }

    public void learning(List<Double> input, double c) {
        sys.learn(input, cr, c);
    }

    public void showWeight() {
        Layer layer = sys.getLayer(1);
        Neuron neuron = layer.getNeuron(0);
        System.out.println(neuron.getWeight());
        System.out.println("----");
        Layer inputLayer = sys.getLayer(0);
        for (int i = 0; i < inputLayer.getNumNuerons(); i++) {
            Neuron neuron2 = inputLayer.getNeuron(i);
            System.out.println("Is fixed : " + neuron2.isFixOutput());
            System.out.println(neuron2.getWeight());
        }
    }

    public void showStatus(List<List<Double>> allInput) {
        for (List<Double> input : allInput) {
            List<Double> r = response(input);
            List<Double> output = cr.apply(input);
            StringBuilder sb = new StringBuilder();
            sb.append(input.toString()).append(" -> ");
            sb.append(r.toString()).append(" / ");
            sb.append(output.toString());
            System.out.println(sb.toString());
        }
        System.out.println();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Random random = new Random(48L);
        List<List<Double>> allInput = Neuron.allInput(3);
        int numInput = allInput.size();
        double c = .1;
        LearningXOR sys = new LearningXOR(random);
        System.out.println(sys.getClass().getCanonicalName());
        int numTrials = 10000;
        for (int t = 0; t < numTrials; t++) {
            int k = random.nextInt(numInput);
            List<Double> input = allInput.get(k);
            List<Double> r = sys.response(input);
            sys.learning(input, c);
            System.out.println(input);
            sys.showStatus(allInput);
        }
        sys.showWeight();    
    }

}

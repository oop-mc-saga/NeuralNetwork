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
public class SimplestWithoutLeaning {

    private final MultiLayer sys;

    public SimplestWithoutLeaning(Random random) {
        int numNeuron = 1;
        DoubleFunction<Double> f = x -> Math.tanh(10 * x);
        DoubleFunction<Double> g = x -> 1. / Math.cosh(10 * x);
        Integer[] nList = {numNeuron};
        List<Integer> numNeurons = Arrays.asList(nList);
        List<DoubleFunction<Double>> functions = new ArrayList<>();
        functions.add(f);
        List<DoubleFunction<Double>> derivatives = new ArrayList<>();
        derivatives.add(g);
        sys = new MultiLayer(3, numNeurons, functions, derivatives,random);
    }

    public List<Double> response(List<Double> input) {
        return sys.response(input);
    }

    public static void main(String args[]) {
        Random random = new Random(48);
        Double[] input = {1., -1., 1.};
        SimplestWithoutLeaning sys = new SimplestWithoutLeaning(random);
        List<Double> r = sys.response(Arrays.asList(input));
        System.out.println(r);
    }
}

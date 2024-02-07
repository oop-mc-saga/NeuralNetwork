package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleFunction;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author tadaki
 */
public class LayerTest {

    private final double w = 10.;
    private final DoubleFunction<Double> function = x -> Math.tanh(x / w);

    public LayerTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of response method, of class Layer.
     */
    @Test
    public void testResponse() {
        System.out.println("response of a single neuron case");
        Double[] inputArray = {1., 1., 1.};
        List<Double> input = Arrays.asList(inputArray);
        // create instance of a layer with one neuron
        Double[] wArray = {1., 1., 1.};
        List<Double> weight = Arrays.asList(wArray);

        Neuron neuron = new Neuron("test", weight, function);
        List<Neuron> neurons = new ArrayList<>();
        neurons.add(neuron);

        Layer instance = new Layer(input.size(), neurons);

        double r = Neuron.product(input, weight);
        List<Double> expResult = new ArrayList<>();
        expResult.add(function.apply(r));

        List<Double> result = instance.response(input);
        assertEquals(expResult, result);
    }

    @Test
    public void testResponse2() {
        System.out.println("response of a two neuron case");
        Double[] inputArray = {1., 1., 1.};
        List<Double> input = Arrays.asList(inputArray);
        List<Neuron> neurons = new ArrayList<>();
        List<Double> expResult = new ArrayList<>();
        // create instance of a layer with one neuron
        {
            Double[] wArray = {1., 1., 1.};
            List<Double> weight = Arrays.asList(wArray);

            Neuron neuron = new Neuron("test", weight, function);
            neurons.add(neuron);
            double r = Neuron.product(input, weight);
            expResult.add(function.apply(r));
        }
        {
            Double[] wArray = {1., -1., 1.};
            List<Double> weight = Arrays.asList(wArray);

            Neuron neuron = new Neuron("test", weight, function);
            neurons.add(neuron);
            double r = Neuron.product(input, weight);
            expResult.add(function.apply(r));
        }

        Layer instance = new Layer(input.size(), neurons);

        List<Double> result = instance.response(input);
        assertEquals(expResult, result);
    }
}

package model;

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
public class NeuronTest {

    private final double w = 10.;
    private final DoubleFunction<Double> function = x -> Math.tanh(x / w);

    public NeuronTest() {
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
     * Test of response method, of class Neuron.
     */
    @Test
    public void testResponse() {
        System.out.println("response");
        Double[] inputArray = {1., 1., 1.};
        List<Double> input = Arrays.asList(inputArray);

        Double[] wArray = {1., 1., 1.};
        List<Double> weight = Arrays.asList(wArray);
        Neuron instance = new Neuron("test", weight, function);
        double r = 0.;
        for (int i = 0; i < weight.size(); i++) {
            r += weight.get(i) * input.get(i);
        }
        Double expResult = function.apply(r);
        Double result = instance.response(input);
        assertEquals(expResult, result);
    }

    /**
     * Test of normalizeWeight method, of class Neuron.
     */
    @Test
    public void testNormalizeWeight() {
        System.out.println("normalizeWeight");
        Double[] wArray = {1., 1., 1.};
        List<Double> weight = Arrays.asList(wArray);
        double delta = 1e-5;
        Neuron instance = new Neuron("test", weight, function);
        instance.normalizeWeight();
        List<Double> normalized = instance.getWeight();
        double x = 0.;
        for (Double d : normalized) {
            x += d * d;
        }
        assertEquals(1, Math.sqrt(x), delta);
    }


}

package model;

import java.util.List;

/**
 * Correct responce of newron for input
 *
 * @author tadaki
 */
public interface CorrectResponse {

    public List<Double> apply(List<Double> input);
}

package neuralnetwork;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class Link {
    private Neuron dest;
    private double weight;

    public Link (Neuron dest, double weight) {
        this.dest = dest;
        this.weight = weight;
    }
}

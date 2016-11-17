package neuralnetwork;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class Link {
    private Neuron dest;
    private Neuron src;
    private double weight;

    public Link (Neuron dest, Neuron src, double weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }

    public Neuron getDest() {
        return dest;
    }

    public void setDest(Neuron dest) {
        this.dest = dest;
    }

    public Neuron getSrc() {
        return src;
    }

    public void setSrc(Neuron src) {
        this.src = src;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    @Override
    public String toString() {
        return "Link: (" + dest.getName() + ") to (" + src.getName() + "), weight: " + weight;
    }
}

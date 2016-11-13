package neuralnetwork;

import java.util.List;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class Neuron {
    private int neuronNumber;
    private int value;
    private Layer layer;
    private List<Link> outLinks;

    public Neuron(int neuronNumber, int value, Layer layer) {
        this.neuronNumber = neuronNumber;
        this.value = value;
        this.layer = layer;
        this.outLinks = null;
    }

    public int getNeuronNumber() {
        return neuronNumber;
    }

    public int getValue() {
        return value;
    }

    public Layer getLayer() {
        return layer;
    }

    public List<Link> getOutLinks() {
        return outLinks;
    }

    public void setNeuronNumber(int neuronNumber) {
        this.neuronNumber = neuronNumber;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public void setLayer(Layer layer) {
        this.layer = layer;
    }

    public void setOutLinks(List<Link> outLinks) {
        this.outLinks = outLinks;
    }
}

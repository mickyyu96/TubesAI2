package neuralnetwork;

import java.util.List;

/**
 * Created by ranggarmaste on 11/13/16.
 */
public class Layer {
    private List<Neuron> neurons;
    private int layerNumber;
    private Layer previousLayer;
    private Layer nextLayer;

    public Layer(int layerNumber) {
        this.layerNumber = layerNumber;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void setNeurons(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    public int getLayerNumber() {
        return layerNumber;
    }

    public void setLayerNumber(int layerNumber) {
        this.layerNumber = layerNumber;
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    @Override
    public String toString() {
        return "Layer " + layerNumber;
    }
}

package neuralnetwork;

import java.util.List;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class Neuron {
    private int neuronNumber;
    private int value;
    private String name;
    private List<Link> next;
    private List<Link> prev;

    public Neuron(int neuronNumber, String name) {
        this.neuronNumber = neuronNumber;
        this.name = name;
    }

    public Neuron(int neuronNumber, int value, String name) {
        this.neuronNumber = neuronNumber;
        this.value = value;
        this.name = name;
    }

    public int getNeuronNumber() {
        return neuronNumber;
    }

    public int getValue() {
        return value;
    }

    public String getName() {
        return name;
    }

    public List<Link> getNext() {
        return next;
    }

    public List<Link> getPrev() {
        return prev;
    }

    public void setNeuronNumber(int neuronNumber) {
        this.neuronNumber = neuronNumber;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setNext(List<Link> next) {
        this.next = next;
    }

    public void setPrev(List<Link> prev) {
        this.prev = prev;
    }

    @Override
    public String toString() {
        return "Neuron: " + name;
    }
}

package neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class Neuron implements Serializable {
    private int neuronNumber;
    private double value;
    private double error;
    private String name;
    private List<Link> next;
    private List<Link> prev;

    public Neuron(int neuronNumber, String name) {
        this.neuronNumber = neuronNumber;
        this.name = name;
        this.next = new ArrayList<>();
        this.prev = new ArrayList<>();
    }

    public Neuron(int neuronNumber, int value, String name) {
        this.neuronNumber = neuronNumber;
        this.value = value;
        this.name = name;
        this.next = new ArrayList<>();
        this.prev = new ArrayList<>();
    }

    public int getNeuronNumber() {
        return neuronNumber;
    }

    public double getValue() {
        return value;
    }

    public double getError() { return error; }

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

    public void setValue(double value) {
        this.value = value;
    }

    public void setError(double error) { this.error = error; }

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

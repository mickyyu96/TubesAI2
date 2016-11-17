import neuralnetwork.Layer;
import neuralnetwork.Link;
import neuralnetwork.Neuron;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.ConverterUtils;

import java.util.*;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class ANN implements Classifier, CapabilitiesHandler {
    private Instances m_Instances;
    private int hiddenNodes;
    private int totalLayers;
    private List<Layer> layers;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_Instances = new Instances(instances);
        layers = new ArrayList<>();
        totalLayers = hiddenNodes == 0 ? 2 : 3;

        // Filter nominal to numeric: til' next time

        // Create input layer
        int layerCount = 0;
        int neuronCount = 0;
        Layer lIn = new Layer(layerCount);
        List<Neuron> neurons = new ArrayList<>();
        Enumeration<Attribute> enu = instances.enumerateAttributes();
        while (enu.hasMoreElements()) {
            Attribute attr = enu.nextElement();
            Neuron neuron = new Neuron(neuronCount, attr.name());
            neurons.add(neuron);
            neuronCount++;
        }
        lIn.setNeurons(neurons);
        layers.add(lIn);

        // Should I create hidden layer?
        if (hiddenNodes > 0) {
            neurons = new ArrayList<>();
            layerCount++;
            Layer lHidden = new Layer(layerCount);
            for (int i = 0; i < hiddenNodes; i++) {
                Neuron neuron = new Neuron(i, "hidden-" + i);
                neurons.add(neuron);
            }
            lHidden.setNeurons(neurons);
            layers.add(lHidden);
        }

        // Create output layer
        layerCount++;
        Layer lOut = new Layer(layerCount);
        neurons = new ArrayList<>();
        neuronCount = 0;
        Enumeration<Object> classVal = instances.classAttribute().enumerateValues();
        while (classVal.hasMoreElements()) {
            String label = (String) classVal.nextElement();
            Neuron neuron = new Neuron(neuronCount, label);
            neurons.add(neuron);
            neuronCount++;
        }
        lOut.setNeurons(neurons);
        layers.add(lOut);

        // Connect neurons
        connectNeurons();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 1.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public Enumeration<Option> listOptions() {
        Vector newVector = new Vector(1);
        newVector.addElement(new Option("\tUse 1 hidden layer and how many nodes.", "H", 1, "-H <no. of nodes>"));
        return newVector.elements();
    }

    public String[] getOptions() {
        Vector options = new Vector();
        if (hiddenNodes > 0) {
            options.add("-H");
            options.add("" + hiddenNodes);
        }
        return (String[])options.toArray(new String[0]);
    }

    public void setOptions(String[] options) throws Exception {
        String nodes = Utils.getOption('H', options);
        if (nodes.length() == 0) {

        } else {
            int totalNodes = Integer.parseInt(nodes);
            if (totalNodes < 0) {
                throw new Exception("Number of hidden nodes cannot be negative.");
            }
            hiddenNodes = totalNodes;
        }
    }

    private double sigmoid(double x) {
        return (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }

    private void connectNeurons() {
        // Connect neurons
        for (Layer l : layers) {
            if (l.getLayerNumber() != 0) {
                Layer prev = layers.get(l.getLayerNumber() - 1);
                for (Neuron n : l.getNeurons()) {
                    List<Link> links = new ArrayList<>();
                    // Add bias link
                    links.add(new Link(new Neuron(-1, 1, "bias"), n, 1));
                    for (Neuron nPrev : prev.getNeurons()) {
                        Link link = new Link(nPrev, n, 1);
                        links.add(link);
                    }
                    n.setPrev(links);
                }
            }
            if (l.getLayerNumber() != totalLayers - 1) {
                Layer next = layers.get(l.getLayerNumber() + 1);
                for (Neuron n : l.getNeurons()) {
                    List<Link> links = new ArrayList<>();
                    for (Neuron nNext : next.getNeurons()) {
                        Link link = new Link(n, nNext, 1);
                        links.add(link);
                    }
                    n.setNext(links);
                }
            }
        }
    }

    public void debugPrint() {
        for (Layer l : layers) {
            System.out.println("==" + l + "==");
            System.out.println("Previous neurons");
            if (l.getLayerNumber() == 0) {
                System.out.println("** No prev neurons **");
            } else {
                for (Neuron n : l.getNeurons()) {
                    System.out.println(n);
                    for (Link link : n.getPrev()) {
                        System.out.println(link);
                    }
                }
            }

            System.out.println("Next neurons");
            if (l.getLayerNumber() == totalLayers - 1) {
                System.out.println("** No next neurons **");
            } else {
                for (Neuron n : l.getNeurons()) {
                    System.out.println(n);
                    for (Link link : n.getNext()) {
                        System.out.println(link);
                    }
                }
            }
        }
    }
}

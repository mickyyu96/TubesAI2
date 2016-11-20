import neuralnetwork.Layer;
import neuralnetwork.Link;
import neuralnetwork.Neuron;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.ConverterUtils;

import java.util.*;

import static java.lang.Math.abs;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class ANN implements Classifier, CapabilitiesHandler {
    private Instances m_Instances;
    private int hiddenNodes;
    private int totalLayers;
    private int maxIterations = 10; // default: 10
    private double errorThreshold = 0.0005; // default: 0
    private double learningRate = 0.01;
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
        connectLayers();
        connectNeurons();

        // Learning
//        feedForward(m_Instances.instance(0));
//        System.out.println(checkError(m_Instances.instance(0)));
//        backPropagate(m_Instances.instance(0));
//        System.out.println();
        for (int i = 0; i < maxIterations; i++) {
            Enumeration<Instance> enuins = m_Instances.enumerateInstances();
            while (enuins.hasMoreElements()) {
                Instance instance = enuins.nextElement();
                feedForward(instance);
                double err = checkError(instance);
                //System.out.println("error = " + err);
                if (abs(err - errorThreshold) <= 0.00001) {
                    break;
                } else {
                    backPropagate(instance);
                }
            }
        }

        debugPrint();
        System.out.println();
        System.out.println(classifyInstance(m_Instances.instance(0)));

//        Enumeration<Instance> enuins = m_Instances.enumerateInstances();
//        while (enuins.hasMoreElements()) {
//            Instance ins = enuins.nextElement();
//            int a = (int) classifyInstance(ins);
//            System.out.println(a);
//        }
    }

    private double sumNet (Neuron neuron) {
        double net = 0;
        List<Link> link = neuron.getPrev();
        for (Link l : link) {
            net += l.getSrc().getValue() * l.getWeight();
        }
        return net;
    }

    private void feedForward(Instance instance) {
        double net;
        Layer layer = layers.get(0);

        //assign input
        int j = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (i != instance.classIndex()) {
                layer.getNeurons().get(j).setValue(instance.value(i));
                System.out.println(layer.getNeurons().get(j).getName() + " = " + layer.getNeurons().get(j).getValue());
                j++;
            }
        }

        //next layer
        while (layer.getNextLayer() != null) {
            layer = layer.getNextLayer();
            for (Neuron n : layer.getNeurons()) {
                net = sumNet(n);
                n.setValue(sigmoid(net));

                System.out.println(n.getName() + " = " + n.getValue());
            }
        }

        System.out.println();
    }

    private double checkError(Instance instance) {
        double err = 0.0;
        double sumerr = 0.0;
        int lastIdx = totalLayers - 1;
        Layer lastL = layers.get(lastIdx);
        double target;

        //sum err output
        //System.out.println();
        //System.out.println(instance.stringValue(instance.classIndex()));
        for (Neuron n : lastL.getNeurons()) {
            if (n.getName() == instance.stringValue(instance.classIndex())) {
                target = 1;
            } else {
                target = 0;
            }
            err = n.getValue() * (1 - n.getValue()) * (target - n.getValue());
            n.setError(err);
            sumerr += err;
        }

        return abs(sumerr);
    }

    private void backPropagate(Instance instance) {
        int lastIdx = totalLayers - 1;
        Layer lastL = layers.get(lastIdx);
        double updateW;

        //update weight to output
        for (Neuron n : lastL.getNeurons()) {
            for (Link l : n.getPrev()) {
                updateW =  l.getWeight() + learningRate * n.getError() * l.getSrc().getValue();
                l.setWeight(updateW);
            }
        }

        //count err if there is a hidden layer
        if (totalLayers == 3) {
            Layer hiddenL = lastL.getPreviousLayer();
            double sigma;
            double err;

            //count err for hidden layer
            for (Neuron n : hiddenL.getNeurons()) {
                sigma = 0;
                //sum weight * err
                for (Link l : n.getNext()) {
                    sigma += l.getWeight() * l.getDest().getError();
                }

                err = n.getValue() * (1 - n.getValue()) * sigma;
                n.setError(err);
            }

            //update weight to hidden layer
            for (Neuron n : hiddenL.getNeurons()) {
                for (Link l : n.getPrev()) {
                    updateW = l.getWeight() + learningRate * n.getError() * l.getSrc().getValue();
                    l.setWeight(updateW);
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        int lastIdx = totalLayers-1;
        Layer lastL = layers.get(lastIdx);
        Neuron max = lastL.getNeurons().get(0);

        feedForward(instance);
        for (Neuron n : lastL.getNeurons()) {
            if (n.getValue() > max.getValue()) {
                max = n;
            }
        }

        return max.getNeuronNumber();
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
        newVector.addElement(new Option("\tSpecify the max iterations/epochs of learning.", "I", 1, "-I <no. of max iterations>"));
        newVector.addElement(new Option("\tSpecify the error threshold of learning.", "E", 1, "-E <error threshold>"));
        return newVector.elements();
    }

    public String[] getOptions() {
        Vector options = new Vector();
        options.add("-H");
        options.add("" + hiddenNodes);
        options.add("-I");
        options.add("" + maxIterations);
        options.add("-E");
        options.add("" + errorThreshold);
        return (String[])options.toArray(new String[0]);
    }

    public void setOptions(String[] options) throws Exception {
        String nodes = Utils.getOption('H', options);
        if (nodes.length() > 0) {
            int totalNodes = Integer.parseInt(nodes);
            if (totalNodes < 0) {
                throw new Exception("Number of hidden nodes cannot be negative.");
            }
            hiddenNodes = totalNodes;
        }
        String iterationsString = Utils.getOption('I', options);
        if (iterationsString.length() > 0) {
            int iterations = Integer.parseInt(iterationsString);
            if (iterations <= 0) {
                throw new Exception("Number of epochs/iteration must be greater than 0");
            }
            maxIterations = iterations;
        }
        String thresholdString = Utils.getOption('E', options);
        if (nodes.length() > 0) {
            double threshold = Double.parseDouble(thresholdString);
            if (threshold < 0) {
                throw new Exception("Threshold cannot be negative.");
            }
            errorThreshold = threshold;
        }
    }

    private double sigmoid(double x) {
        return (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }

    private void connectLayers() {
        for (int i = 0; i < totalLayers - 1; i++) {
            Layer firstLayer = layers.get(i);
            Layer secondLayer = layers.get(i + 1);
            firstLayer.setNextLayer(secondLayer);
            secondLayer.setPreviousLayer(firstLayer);
        }
    }
    
    private void connectNeurons() {
        // Connect neurons
        for (Layer l : layers) {
            Layer next = l.getNextLayer();
            if (next == null) continue;

            for (Neuron n : l.getNeurons()) {
                for (Neuron nNext : next.getNeurons()) {
                    Link link = new Link(n, nNext, 1);
                    n.getNext().add(link);
                    nNext.getPrev().add(link);
                }
            }
            Neuron biasNeuron = new Neuron(0, 1, "bias");
            for (Neuron nNext : next.getNeurons()) {
                nNext.getPrev().add(new Link(biasNeuron, nNext, 1));
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

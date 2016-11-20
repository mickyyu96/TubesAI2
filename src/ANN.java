import neuralnetwork.Layer;
import neuralnetwork.Link;
import neuralnetwork.Neuron;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class ANN extends AbstractClassifier implements CapabilitiesHandler {
    private Instances m_Instances;
    private Instances filteredInstances;
    private Filter nomToBinFilter;
    private Filter normalizeFilter;
    private Filter replaceFilter;
    private int hiddenNodes;
    private int totalLayers;
    private int maxIterations = 1000;
    private double errorThreshold = 0.00005;
    private double learningRate = 0.5;
    private List<Layer> layers;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_Instances = new Instances(instances);
        layers = new ArrayList<>();
        totalLayers = hiddenNodes == 0 ? 2 : 3;

        // Filter normalize
        Normalize normalize = new Normalize();
        normalize.setInputFormat(instances);
        normalizeFilter = normalize;
        filteredInstances = Filter.useFilter(instances, normalize);

        // Filter nominal to numeric
        NominalToBinary nomToBin = new NominalToBinary();
        nomToBin.setInputFormat(filteredInstances);
        nomToBinFilter = nomToBin;
        filteredInstances = Filter.useFilter(filteredInstances, nomToBin);

        // Replace missing values
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(filteredInstances);
        replaceFilter = replaceMissing;
        filteredInstances = Filter.useFilter(filteredInstances, replaceMissing);

        // Create input layer
        int layerCount = 0;
        int neuronCount = 0;
        Layer lIn = new Layer(layerCount);
        List<Neuron> neurons = new ArrayList<>();
        Enumeration<Attribute> enu = filteredInstances.enumerateAttributes();
        while (enu.hasMoreElements()) {
            Attribute attr = enu.nextElement();
            //System.out.println(attr.name());
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
        Enumeration<Object> classVal = filteredInstances.classAttribute().enumerateValues();

        // Count how many classVal?
        int numClass = 0;
        while (classVal.hasMoreElements()) {
            classVal.nextElement();
            numClass++;
        }
        classVal = filteredInstances.classAttribute().enumerateValues();
        if (numClass > 2) {
            while (classVal.hasMoreElements()) {
                String label = (String) classVal.nextElement();
                Neuron neuron = new Neuron(neuronCount, label);
                neurons.add(neuron);
                neuronCount++;
            }
        } else {
            Neuron neuron = new Neuron(neuronCount, "class");
            neurons.add(neuron);
        }
        lOut.setNeurons(neurons);
        layers.add(lOut);

        // Connect layers, neurons, and initialize weights
        connectLayers();
        connectNeurons();
        initializeWeights();

        for (int i = 0; i < maxIterations; i++) {
            System.out.println("Iteration-" + i);
            Enumeration<Instance> enuins = filteredInstances.enumerateInstances();
            while (enuins.hasMoreElements()) {
                Instance instance = enuins.nextElement();
                feedForward(instance);
                checkError(instance);
                backPropagate(instance);
            }
            enuins = filteredInstances.enumerateInstances();

            double errorTotal = 0.0;
            while (enuins.hasMoreElements()) {
                Instance instance = enuins.nextElement();
                feedForward(instance);
                errorTotal += checkError(instance);
            }
        }
        //debugPrint();
        //System.out.println();

        Enumeration<Instance> enuins = filteredInstances.enumerateInstances();
        int total = 0;
        int same = 0;
        while (enuins.hasMoreElements()) {
            Instance ins = enuins.nextElement();
            int a = (int) classifyInstance(ins);
            //System.out.println(a);
            total++;
            if (a == (int) ins.classValue()) {
                same++;
            }
        }
    }

    public Instances getFilteredInstances() {
        return filteredInstances;
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
                //System.out.println(layer.getNeurons().get(j).getName() + " = " + layer.getNeurons().get(j).getValue());
                j++;
            }
        }

        //next layer
        while (layer.getNextLayer() != null) {
            layer = layer.getNextLayer();
            for (Neuron n : layer.getNeurons()) {
                net = sumNet(n);
                n.setValue(sigmoid(net));

                //System.out.println(n.getName() + " = " + n.getValue());
            }
        }

        //System.out.println();
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
        if (lastL.getNeurons().size() == 1) {
            Neuron n = lastL.getNeurons().get(0);
            target = instance.classValue();
            err = n.getValue() * (1 - n.getValue()) * (target - n.getValue());
            n.setError(err);
            sumerr += abs(err);
        } else {
            for (Neuron n : lastL.getNeurons()) {
                if (n.getName() == instance.stringValue(instance.classIndex())) {
                    target = 1;
                } else {
                    target = 0;
                }
                err = n.getValue() * (1 - n.getValue()) * (target - n.getValue());
                n.setError(err);
                sumerr += abs(err);
            }
        }
        return sumerr;
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
        if (lastL.getNeurons().size() == 1) {
            Neuron n = lastL.getNeurons().get(0);
            if (n.getValue() < 0.5) {
                return 0;
            } else {
                return 1;
            }
        } else {
            for (Neuron n : lastL.getNeurons()) {
                if (n.getValue() > max.getValue()) {
                    max = n;
                }
            }
            return max.getNeuronNumber();
        }
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
        newVector.addElement(new Option("\tSpecify the learning rate.", "L", 1, "-L <learning rate>"));
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
        options.add("-L");
        options.add("" + learningRate);
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
        String learnString = Utils.getOption('L', options);
        if (nodes.length() > 0) {
            double learn = Double.parseDouble(learnString);
            if (learn < 0) {
                throw new Exception("Learning rate cannot be negative.");
            }
            learningRate = learn;
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
                nNext.getPrev().add(new Link(biasNeuron, nNext, 0));
            }
        }
    }

    private void initializeWeights() {
        Random rand = new Random();
        rand.setSeed(123456);
        for (Layer l : layers) {
            if (l.getLayerNumber() == 0) continue; // skip first layer
            for (Neuron n : l.getNeurons()) {
                double std = sqrt(2.0 / (l.getPreviousLayer().getNeurons().size() + l.getNeurons().size()));
                for (Link link : n.getPrev()) {
                    if (!link.equals("bias")) {
                        double weight = rand.nextGaussian() * std;
                        link.setWeight(weight);
                    }
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

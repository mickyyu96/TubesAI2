package neuralnetwork;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by ranggarmaste on 11/13/16.
 */

public class ANN implements Classifier {
    private Instances m_Instances;
    private int m_nAttributes;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_Instances = instances;

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}

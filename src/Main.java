import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.util.Enumeration;
import java.util.Random;

/**
 * Created by ranggarmaste on 11/16/16.
 */
public class Main {
    public static void main(String[] args) {
        Instances data = null;
        try {
            data = ConverterUtils.DataSource.read("data/Team.arff");
            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }

        ANN ann = new ANN();
        String[] options = new String[8];
        options[0] = "-H";
        options[1] = "0"; // no. of hidden nodes. Set to 0 if hidden layers isn't needed
        options[2] = "-I";
        options[3] = "100000"; // max iterations/epochs
        options[4] = "-E";
        options[5] = "0.1"; // error threshold
        options[6] = "-L";
        options[7] = "0.1"; // learning rate
        try {
            ann.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            ann.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println(data);
        System.out.println(ann.getFilteredInstances());
        // Save Model
        try {
            SerializationHelper.write("ann.model", ann);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Load Model
        try {
            Object obj[] = SerializationHelper.readAll("ann.model");
            ann = (ANN) obj[0];
        } catch (Exception e) {
            e.printStackTrace();
        }

        // K-FOLD EVALUATION
        Evaluation eval = null;
        try {
            eval = new Evaluation(ann.getFilteredInstances());
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            eval.crossValidateModel(ann, ann.getFilteredInstances(), 10, new Random(1));
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println();
        System.out.println("**** 10-Fold Cross Validation Evaluation ****");
        System.out.println(eval.toSummaryString("\nResults\n", false));;
        try {
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }

        // FULL TRAINING SET
        try {
            eval = new Evaluation(ann.getFilteredInstances());
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            eval.evaluateModel(ann, ann.getFilteredInstances());
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("**** Full Training Set Evaluation ****");
        System.out.println(eval.toSummaryString("\nResults\n", false));
        try {
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

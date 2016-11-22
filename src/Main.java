import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Enumeration;
import java.util.Random;
import java.util.StringJoiner;

/**
 * Created by ranggarmaste on 11/16/16.
 */
public class Main {
    public static void evalKFold(ANN ann, Instances data, int folds) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(ann, data, 10, new Random(1));
        System.out.println("**** " + folds + "-Fold Cross Validation Evaluation ****");
        System.out.println(eval.toSummaryString("\nResults\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public static void evalDataTrain(ANN ann, Instances data, String title) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(ann, data);
        if (title.equals("train")) {
            System.out.println("**** Full Training Set Evaluation ****");
        } else {
            System.out.println("**** Test Set Evaluation ****");
        }
        System.out.println(eval.toSummaryString("\nResults\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public static void evalSplit(ANN ann, Instances data, double trainPercent) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * trainPercent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(ann, test);
        System.out.println("**** Split Test Evaluation: " + trainPercent + "% Train Set ****");
        System.out.println(eval.toSummaryString("\nResults\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    public static void saveModel(Classifier ann, String filename) throws Exception {
        SerializationHelper.write(filename, ann);
    }

    public static Classifier loadModel(String filename) throws Exception {
        Object obj[] = SerializationHelper.readAll("ann.model");
        return (Classifier) obj[0];
    }

    public static void main(String[] args) {
        Instances data = null;
        try {
            data = ConverterUtils.DataSource.read("data/Team.arff");
            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }

        ANN ann = new ANN();

        /*** USER INPUT ***/

        double percent = 60; // Split test percentage
        String[] options = new String[10];
        options[0] = "-H";
        options[1] = "25"; // no. of hidden nodes. Set to 0 if hidden layers isn't needed
        options[2] = "-I";
        options[3] = "100"; // max iterations/epochs
        options[4] = "-E";
        options[5] = "0"; // error threshold
        options[6] = "-L";
        options[7] = "1"; // learning rate
        options[8] = "-F";
        options[9] = "N"; // S = Standardize, N = Normalize, X = No Filter

        /*** END OF USER INPUT ***/

        /** FILTER **/
        Instances filteredData = new Instances(data);
        NominalToBinary nomToBin = null;
        Filter numericFilter = null;
        ReplaceMissingValues replaceMissing = null;
        try {
            // Filter normalize
            String numericFilterType = options[9];
            if (!numericFilterType.equals("X")) {

                if (numericFilterType.equals("N")) {
                    numericFilter = new Normalize();
                } else if (numericFilterType.equals("S")) {
                    numericFilter = new Standardize();
                }
                numericFilter.setInputFormat(data);
                filteredData = Filter.useFilter(filteredData, numericFilter);
            }

            // Filter nominal to binary
            nomToBin = new NominalToBinary();
            nomToBin.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, nomToBin);

            // Filter nominal to numeric

            // Replace missing values
            replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // SET OPTIONS + BUILD CLASSIFIER
        try {
            ann.setOptions(options);
            ann.buildClassifier(filteredData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // SAVING MODEL & LOAD MODEL
        String fileName = "ann.model";
        try {
            saveModel(ann, fileName);
            ann = (ANN) loadModel(fileName);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // EVALUATIONS
        Instances testSet = null;
        try {
            testSet = ConverterUtils.DataSource.read("data/Team_test.arff");
            testSet.setClassIndex(testSet.numAttributes() - 1);
            Filter.useFilter(testSet, numericFilter);
            Filter.useFilter(testSet, nomToBin);
            Filter.useFilter(testSet, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            System.out.println();
            evalKFold(ann, filteredData, 10);
            evalDataTrain(ann, filteredData, "train");
            //evalSplit(ann, filteredData, percent);
            evalDataTrain(ann, testSet, "test");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

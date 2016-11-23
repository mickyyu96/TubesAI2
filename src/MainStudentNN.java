import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import java.util.Scanner;

/**
 * Created by ranggarmaste on 11/22/16.
 */

public class MainStudentNN {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("+=== KEPIKEPIKEPI FFNN ===+");
        System.out.println("CHOOSE YOUR CLASS");
        System.out.println("1. Make Model: Dalc");
        System.out.println("2. Make Model: Walc");
        System.out.println("3. Load Model: Dalc");
        System.out.println("4. Load Model: Walc");
        System.out.println("Input 1/2/3/4");
        System.out.println();
        System.out.print("> ");
        int input = sc.nextInt();
        String deletedClassIndex = null;
        if (input == 1 || input == 3) {
            deletedClassIndex = "28";
        } else if (input == 2 || input == 4) {
            deletedClassIndex = "27";
        }

        Instances data = null;
        try {
            data = ConverterUtils.DataSource.read("data/student-train.arff");
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
        options[3] = "1000"; // max iterations/epochs
        options[4] = "-E";
        options[5] = "0"; // error threshold
        options[6] = "-L";
        options[7] = "0.2"; // learning rate
        options[8] = "-F";
        options[9] = "N";

        /*** END OF USER INPUT ***/

        /** FILTER **/
        Remove remove = null;
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

            // Filter remove
            remove = new Remove();
            String[] removeOptions = new String[2];
            removeOptions[0] = "-R"; // "range"
            removeOptions[1] = deletedClassIndex;
            remove.setOptions(removeOptions);
            remove.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, remove);

            // Set class index
            filteredData.setClassIndex(26);

            // Filter nominal to binary
            nomToBin = new NominalToBinary();
            nomToBin.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, nomToBin);

            // Replace missing values
            replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // MODEL NAME
        String fileName = input == 1 || input == 3 ? "dalc-ann.model" : "walc-ann.model";

        if (input == 1 || input == 2) {
            // SET OPTIONS + BUILD CLASSIFIER
            try {
                ann.setOptions(options);
                ann.buildClassifier(filteredData);
                Main.saveModel(ann, fileName);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if (input == 3 || input == 4) {
            try {
                ann = (ANN) Main.loadModel(fileName);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // EVALUATIONS
        Instances testSet = null;
        try {
            testSet = ConverterUtils.DataSource.read("data/student-mat-test.arff");
            if (numericFilter != null) {
                testSet = Filter.useFilter(testSet, numericFilter);
            }
            testSet = Filter.useFilter(testSet, remove);
            testSet.setClassIndex(26);
            testSet = Filter.useFilter(testSet, nomToBin);
            testSet = Filter.useFilter(testSet, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println();
            //Main.evalKFold(ann, filteredData, 10);
            Main.evalDataTrain(ann, filteredData, "train");
            //Main.evalSplit(ann, filteredData, percent);
            Main.evalDataTrain(ann, testSet, "test");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

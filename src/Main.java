import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Created by ranggarmaste on 11/16/16.
 */
public class Main {
    public static void main(String[] args) {
        Instances data = null;
        try {
            data = ConverterUtils.DataSource.read("data/iris.arff");
            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }

        ANN ann = new ANN();
        String[] options = new String[6];
        options[0] = "-H";
        options[1] = "2"; // no. of hidden nodes. Set to 0 if hidden layers isn't needed
        options[2] = "-I";
        options[3] = "10"; // max iterations/epochs
        options[4] = "-E";
        options[5] = "0.1"; // error threshold
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
    }
}

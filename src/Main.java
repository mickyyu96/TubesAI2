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
        String[] options = new String[2];
        options[0] = "-H";
        options[1] = "2";
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

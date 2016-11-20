import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Enumeration;

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

        //System.out.println(data);
        /*** AWAL TEST. ENTAR HAPUS AJA ***/
        // Ambil instance
        Enumeration<Instance> enu = data.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance inst = enu.nextElement();
            for (int i = 0; i < inst.numAttributes(); i++) {
                if (i != inst.classIndex()) {
                    double val = inst.value(i); // Dapat nilai atribut ke-i
                }
            }
        }
        // Mau ambil nama atribut
        /*** AKHIR TEST ***/

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

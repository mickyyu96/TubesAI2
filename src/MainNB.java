import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Scanner;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.FileInputStream;

public class MainNB {
	private Scanner input;

	public Instances ReadArff(String filename) throws Exception {
	    BufferedReader reader = new BufferedReader(
	                             new FileReader("data/" + filename));
	    Instances data = new Instances(reader);
	    data.setClassIndex(data.numAttributes() - 1);
	    reader.close();
	    
	    return data;
	}
	
	public Instances Discretize(Instances data) throws Exception {
		Discretize filter = new Discretize();
		Instances newData;
		
		filter.setInputFormat(data);
		newData = Filter.useFilter(data, filter);
		
		return newData;
	}
	
	public Classifier TenFoldsCrossValidation(Instances data, int idxClass) throws Exception{
		Classifier nb = new NaiveBayes();
		((NaiveBayes) nb).setKelas(idxClass);
		nb.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(nb, data, 10, new Random(1));
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return nb;
	}
	
	public Classifier SplitTest(Instances data, int percent, int idxClass) throws Exception {
		int trainSize = (int) Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		Classifier nb = new NaiveBayes();
		((NaiveBayes) nb).setKelas(idxClass);
		nb.buildClassifier(train);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(nb, test);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return nb;
	}
	
	public Classifier FullTrainingSchema(Instances data, int idxClass) throws Exception{
		Classifier nb = new NaiveBayes();
		((NaiveBayes) nb).setKelas(idxClass);
		nb.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(nb, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return nb;
	}
	
	public void saveModel(String filename, Classifier cls) throws Exception {
		ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(filename));
		output.writeObject(cls);
		output.flush();
		output.close();
	}
	
	public Classifier loadModel(String filename) throws Exception{
		ObjectInputStream fileinput = new ObjectInputStream(new FileInputStream(filename));
		Classifier cls = (Classifier) fileinput.readObject();
		fileinput.close();
		return cls;
	}
	
	public void classifyData(Classifier model, Instances data) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(model, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception{
		MainNB mainNB = new MainNB();
		Scanner input = new Scanner(System.in);
		
		System.out.println("==============================");
		System.out.println("===       Tubes AI 2       ===");
		System.out.println("==============================");
		System.out.println();
		
		int pilihan;
		do {
			System.out.println("Menu: 1. Mengolah dataset");
			System.out.println("      2. Membaca model dan mengklasifikasi Instances");
			System.out.println("      3. Exit");
			System.out.print("Masukkan pilihan: ");
			pilihan = input.nextInt();
			
			if (pilihan == 1) {
				System.out.print("Masukkan file dataset: ");
				String filename = input.next();

				System.out.println("\nMembaca " + filename + "...");
				Instances data = mainNB.ReadArff(filename);
				
				System.out.print("Masukkan indeks kelas: ");
				int idxClass = input.nextInt();
				data.setClassIndex(--idxClass);
				
			    System.out.println("\nHeader dataset:\n");
			    System.out.println(new Instances(data, 0));
			    
			    int pilihan2;
				do {
					System.out.println("Menu: 1. Melakukan Discretize pada data");
				    System.out.println("      2. Melakukan pembelajaran dataset dengan metode 10-fold cross validation");
					System.out.println("      3. Melakukan pembelajaran dataset dengan metode split test");
					System.out.println("      4. Melakukan pembelajaran dataset dengan metode full-training");
					System.out.println("      5. Back");
					System.out.print("Masukkan pilihan: ");
					pilihan2 = input.nextInt();
					
					Classifier cls = new NaiveBayes();
					if (pilihan2 == 1) {
						data = mainNB.Discretize(data);
						System.out.println("\nHeader dataset setelah filter:\n");
					    System.out.println(new Instances(data, 0));
					}
					else if (pilihan2 == 2) {
						cls = mainNB.TenFoldsCrossValidation(data, idxClass + 1);
					}
					else if (pilihan2 == 3) {
						System.out.print("Masukkan persentase split: ");
						int percent = input.nextInt();
						cls = mainNB.SplitTest(data, percent, idxClass + 1);
					}
					else if (pilihan2 == 4) {
						cls = mainNB.FullTrainingSchema(data, idxClass + 1);
					}
					else if (pilihan2 == 5) {
						System.out.println();
					}
					if (pilihan2 == 2 || pilihan2 == 3 || pilihan2 == 4 || pilihan2 == 6){
						System.out.println("Save model pembelajaran? (y/n)");
						System.out.print("Masukkan pilihan: ");
						char answer = (char) System.in.read();
						if(answer == 'y'){
							System.out.print("Masukkan destinasi penyimpanan: ");
							filename = input.next();
							mainNB.saveModel(filename, cls);
							System.out.println("Model berhasil disimpan pada "+filename);
						}
						System.out.println();
					}
				} while (pilihan2 != 5);
			}
			else if (pilihan == 2) {
				System.out.print("Masukkan file model: ");
				String filename;
				filename = input.next();
				System.out.println("\nMembaca model...\n");
				Classifier model = mainNB.loadModel(filename);
			    System.out.println(model);
			    
			    System.out.print("Masukkan file dataset: ");
				String testFile = input.next();
				
				System.out.println("\nMembaca " + testFile + "...");
				Instances data = mainNB.ReadArff(testFile);
				
				System.out.print("Masukkan indeks kelas: ");
				int idxClass = input.nextInt();
				data.setClassIndex(--idxClass);
				
				mainNB.classifyData(model, data);
			}
			else if (pilihan == 3) {
				input.close();
			}
		} while (pilihan != 3);
	}
}

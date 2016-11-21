import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

import java.io.Serializable;
import java.util.Enumeration;

public class NaiveBayes implements Classifier, Serializable {

	int[][][] table;
	double[][][] table_count;
	double[] c;
	private static Instance instance;
	int countAtt;
	int kelas = 999;
	
	public void setKelas(int idx) {
		kelas = idx;
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		int countRow = data.numInstances();
		System.out.println("Jumlah instance: " + countRow);
		countAtt = data.numAttributes();
		System.out.println("Jumlah atribut: " + countAtt);
		int countClasses = data.numClasses();
		System.out.println("Jumlah class: " + countClasses);
		Enumeration<Attribute> numAtts = data.enumerateAttributes();
		int[] countUnique = new int[countAtt];
		for (int i=0; i<countAtt; i++) {
			countUnique[i] = data.numDistinctValues(data.attribute(i));
			System.out.print(countUnique[i] + " ");
		}
		
		System.out.println();
		table = new int[countAtt][countClasses][20];
		table_count = new double[countAtt][countClasses][20];
		for (int j=0; j<countRow; j++) {
		}
		// Compute counts
	    Enumeration<Instance> enumInsts = data.enumerateInstances();
	    int a = 0;
	    while (enumInsts.hasMoreElements()) {
	      Instance instance = enumInsts.nextElement();
	      //System.out.println(a++ + " " + instance.value(4));
	      Enumeration<Attribute> enumAtts = data.enumerateAttributes();
	      int attIndex = 0;
	      while (enumAtts.hasMoreElements()) {
	    	  if (attIndex == kelas - 1) {
	    		  attIndex++;
	    	  }
	    	  else {
		    	  Attribute attribute = enumAtts.nextElement();
		    	  table[attIndex][(int) instance.classValue()][(int)instance.value(attribute)]++;
		    	  //table[attIndex][(int) instance.classValue()].addValue(instance.value(attribute), instance.weight());
		    	  //System.out.println(attribute);
		    	  attIndex++;
	    	  }
	      }
	    }
	    System.out.println();

	    System.out.println("------------------");
	    System.out.println("Tabel Probabilitas");
	    System.out.println("------------------");
    	int classValue = 0;
    	double sum = 0;
    	c = new double[countClasses];
	    while (classValue < countClasses) {
	    int attIdx = 0;
	    Enumeration<Attribute> enu = data.enumerateAttributes();
	    	while (enu.hasMoreElements()) {
	    	  
	    	  if (attIdx == kelas-1) {
	    		  attIdx++;
	    	  }
	    	  else {
	    	      Attribute attribute = enu.nextElement();
	    	      //System.out.println(attribute);
	    	      int attValue = 0;
	    	      sum = 0;
	    	      while(attValue < attribute.numValues()) {
	    	    	 sum += table[attIdx][classValue][attValue];
	    	    	 attValue++;
		    	  }
	    	      attValue = 0;
	    	      while(attValue < attribute.numValues()) {
	    	    	 // System.out.println("table["+attIdx+"]["+classValue+"]["+attValue+"]: "+table[attIdx][classValue][attValue]+"/"+sum);
	    	    	  table_count[attIdx][classValue][attValue] = (double) table[attIdx][classValue][attValue] / (double) sum;
	    	    	  //System.out.println(table_count[attIdx][classValue][attValue]);
	    	    	  attValue++;
		    	 }
		    	 attIdx++;
	    	  }
	       }
	    System.out.println("Probabilitas kelas ke-" + classValue + ": " + sum + "/" + countRow);
	      c[classValue] = (double) sum / (double) countRow;
	    classValue++;
	    }
	    
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double ans = 0;
		System.out.println("loooooooool: "+ans);
		double[] cd = distributionForInstance(instance);
		double sum = cd[0];
		int cls = 1;
		while(cls < c.length){ 
			if(sum < cd[cls]){
				sum = cd[cls];
				ans = cls;
			}
			cls++;
		}
		System.out.println("loooooooool: "+ans);
		return ans;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		//System.out.println(instance);
		double[] cd = new double[c.length];
		int cls = 0;
		while(cls < c.length){ 
			int attValue = 0;
			double sum = c[cls];
			//System.out.print("sum="+sum+" ");
		    while(attValue < countAtt) {
		    	
		    	if (attValue == kelas - 1) {
		    		attValue++;
		    	}
		    	else {
			    	//System.out.print("1: " + attValue + " ");
			    	//System.out.print("2: " + cls + " ");
			    	//System.out.print("3: " + (int)instance.value(attValue) + " ");
				   	sum *= table_count[attValue][cls][(int)instance.value(attValue)];
				   	//System.out.println("*"+table_count[attValue][cls][(int)instance.value(attValue)]);
				   	attValue++;
		    	}
		    }
		    //System.out.println();
		    cd[cls] = sum;
			//System.out.println("kelas-"+cls+":"+sum);
		    cls++;
		}
		//System.out.println();
		return cd;
	}

	@Override
	public Capabilities getCapabilities() {
		/*Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable( Capability.MISSING_VALUES );

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.MISSING_CLASS_VALUES);

	    // instances
	    result.setMinimumNumberInstances(0);*/

	    return null;
	}
}

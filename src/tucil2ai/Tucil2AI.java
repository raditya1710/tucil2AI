/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package tucil2ai;
import java.io.File;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.core.SerializationHelper;
/**
 *
 * @author t420s
 */

public class Tucil2AI {
    
    /**
     *
     * @param cls
     * @throws Exception
     */
    protected static void saveModel(Classifier cls) throws Exception{
         weka.core.SerializationHelper.write("res/tucilAI2j48.model", cls);
    }
    
    /**
     *
     * @return
     * @throws Exception
     */
    protected static Classifier loadModel() throws Exception{
        return (Classifier) weka.core.SerializationHelper.read("res/tucilAI2j48.model");
    }
    
     /**
     * @param filename
     * @param f
     * @return 
     * @throws java.lang.Exception
     */
    protected static Instances loadfile(String filename, Discretize f) throws Exception{
        DataSource source = new DataSource(filename);
        Instances data;
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
  
        Instances data_filtered = Filter.useFilter(data, f);
        return data_filtered;
    }
    /**
     *
     * @param cls
     * @param data
     * @param cross
     * @return
     * @throws Exception
     */
    protected static Evaluation evalJ48(Classifier cls, Instances data, boolean cross) throws Exception{
        Evaluation E;
        E = new Evaluation(data);
        if(cross == false){
            E.evaluateModel(cls, data);
        }
        else{
            E.crossValidateModel(cls, data, 10, new Random(0x100)); /*crossValidateModel*/
        }
        return E;
    }
    
    /**
     *
     * @param E
     * @throws Exception
     */
    public static void printEval(Evaluation E) throws Exception{
        System.out.println(E.toSummaryString("\nResults\n======\n", false));
        System.out.println(E.toClassDetailsString());
        System.out.println(E.toMatrixString());
    }
    
    /**
     *
     * @param cls
     * @param filename
     * @param f
     * @throws Exception
     */
    public static void ClassifyJ48(Classifier cls, String filename, Discretize f) throws Exception{
        Instances unlabeled = loadfile(filename, f);
        Instances labeled = new Instances(unlabeled);
        for (int i = 0;i < unlabeled.numInstances(); ++i){
            double clsLabel = cls.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        
        System.out.println(labeled.toString());
    }
    
    /**
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        DataSource source;
        Instances data;
        Instances data_filtered;
        Discretize filter;
        
        // input files
        source = new DataSource("res/iris.arff");
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        // filter
        filter = new Discretize();
        filter.setInputFormat(data);
        data_filtered = Filter.useFilter(data, filter);
       
        Classifier clsJ48 = new J48();
        clsJ48.buildClassifier(data_filtered);
        //Evaluation with J48 Decision Tree
        Evaluation eval = evalJ48(clsJ48, data_filtered, false);
        printEval(eval);
        
        /*
        File directory = new File(".");
        System.out.println(directory.getCanonicalPath());
        */
        ClassifyJ48(clsJ48, "res/datatest.arff", filter);
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package tucil2ai;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author t420s
 */

public class Tucil2AI {
    /**
     * @param filename
     * @return 
     * @throws java.lang.Exception
     */
    protected static Instances load(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        Instances data;
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    /**
     *
     * @param data
     * @param cross
     * @return
     * @throws Exception
     */
    protected static Evaluation evalJ48(Instances data, boolean cross) throws Exception{
        Evaluation E;
        E = new Evaluation(data);
        if(cross == false){
            Classifier clsJ48 = new J48();
            clsJ48.buildClassifier(data);
            E.evaluateModel(clsJ48, data);
        }
        else{
            E.crossValidateModel(new J48(), data, 10, new Random(0x1)); /*crossValidateModel*/
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
        source = new DataSource("C:/Program Files/Weka-3-8/data/iris.arff");
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        // filter
        filter = new Discretize();
        filter.setInputFormat(data);
        data_filtered = Filter.useFilter(data, filter);
        
        //Evaluation with J48 Decision Tree
        Evaluation eval = evalJ48(data_filtered, true);
        
        printEval(eval);
    }
}

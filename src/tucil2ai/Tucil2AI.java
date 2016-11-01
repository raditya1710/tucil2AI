/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package tucil2ai;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
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
     * @param args the command line arguments
     */
    
    /*public static void TenFoldsCV(Instances data) throws Exception{
        int seed = 20071996; //bisa diganti sama run
        int folds = 10;
        Random rand = new Random(seed);
        Instances randData = new Instances(data);
        randData.randomize(rand);
        if (randData.classAttribute().isNominal())
            randData.stratify(folds);
        
        Evaluation eval = new Evaluation(randData);
        for(int i = 0;i < folds; ++i){
            Instances train = randData.trainCV(folds, i);
            Instances test = randData.testCV(folds, i);
        }
        
    }*/
    protected static Instances load(String filename) throws Exception{
        DataSource source = new DataSource(filename);
        Instances data;
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        DataSource source;
        Instances data;
        Instances data_filtered;
        Discretize filter;
        source = new DataSource("C:/Program Files/Weka-3-8/data/iris.arff");
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        // filter
        filter = new Discretize();
        filter.setInputFormat(data);
        data_filtered = Filter.useFilter(data, filter);
        /*System.out.println(data.toString());
        System.out.println();
        System.out.println(data_filtered.toString());
        */
        
        Classifier clsJ48 = new J48();
        clsJ48.buildClassifier(data_filtered);
        Evaluation evalJ48 = new Evaluation(data_filtered);
        
        evalJ48.evaluateModel(clsJ48, data_filtered);
        
        
        System.out.println(evalJ48.toSummaryString("\nResults\n======\n", false));
        
    }
}

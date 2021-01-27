import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;

import java.io.IOException;

public class ReviewClassifier {
    public static void main(String[] args) throws Exception {
//        InstanceLoader.printInstances();
        Instances instances = InstanceLoader.readInstances(5);
        Classifier naiveBayesMultinomial = new NaiveBayesMultinomial();
        DataSetClassifier.classify(naiveBayesMultinomial, instances);
    }
}

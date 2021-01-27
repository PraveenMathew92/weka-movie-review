import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.Instances;

import java.io.IOException;

public class ReviewClassifier {
    public static void main(String[] args) throws Exception {
//        InstanceLoader.printInstances();
        Instances instances = InstanceLoader.readInstances(15);
        Classifier naiveBayesMultinomial = new NaiveBayesMultinomial();
        Classifier naiveBayes = new NaiveBayes();
        Classifier naiveBayesMultinomialText = new NaiveBayesMultinomialText();
        DataSetClassifier.classify(naiveBayesMultinomialText, instances);
    }
}

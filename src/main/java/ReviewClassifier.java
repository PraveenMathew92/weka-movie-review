import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ReviewClassifier {
    public static void main(String[] args) throws Exception {
//        InstanceLoader.printInstances();
        Instances instances = InstanceLoader.readInstances(15);
        Classifier naiveBayesMultinomial = new NaiveBayesMultinomial();
        Classifier naiveBayes = new NaiveBayes();
        Classifier naiveBayesMultinomialText = new NaiveBayesMultinomialText();
        Classifier j48 = new J48();
        Classifier linearRegression = new LinearRegression();
        Classifier logisticRegression = new Logistic();
        Classifier smo = new SMOreg();
        DataSetClassifier.classify(logisticRegression, instances);
    }
}

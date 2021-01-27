import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.stopwords.StopwordsHandler;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Enumeration;
import java.util.Random;

public class DataSetClassifier {
    static void classify(Classifier classifier, Instances instances) throws Exception {
        int foldsNumber = 10;

        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(instances);
        filter.setStopwordsHandler(new Rainbow());
        filter.setStemmer(new LovinsStemmer());
        filter.setLowerCaseTokens(true);
        filter.setAttributeIndices("last");

        FilteredClassifier filteredClassifier = new FilteredClassifier();
        filteredClassifier.setClassifier(classifier);
        filteredClassifier.setFilter(filter);

        filteredClassifier.buildClassifier(instances);

        Evaluation evaluation = new Evaluation(instances);
        evaluation.evaluateModel(classifier, instances);
        String classifierName = classifier.getClass().getSimpleName();
        evaluation.crossValidateModel(classifier, instances, foldsNumber, new Random());
        System.out.println(evaluation.toSummaryString(classifierName, true));

        Enumeration<Instance> enumeratedInstances = instances.enumerateInstances();

        while(enumeratedInstances.hasMoreElements()) {
            Instance instance = enumeratedInstances.nextElement();
            System.out.println(instance.toString());
            System.out.println(instances.classAttribute().value((int) filteredClassifier.classifyInstance(instance)));
        }

        System.out.println(evaluation.toMatrixString("CONFUSION MATRIX"));
    }
}

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.stopwords.StopwordsHandler;
import weka.core.tokenizers.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class DataSetClassifier {
    static void classify(Classifier classifier, Instances instances) throws Exception {
        int foldsNumber = 10;

        StringToWordVector filter = new StringToWordVector();

        FilteredClassifier filteredClassifier = new FilteredClassifier();
        filteredClassifier.setFilter(filter);
        filteredClassifier.setClassifier(classifier);

        filteredClassifier.buildClassifier(instances);

        Evaluation evaluation = new Evaluation(Filter.useFilter(instances, filter));
        evaluation.evaluateModel(filteredClassifier, Filter.useFilter(instances, filter));
        String classifierName = filteredClassifier.getClass().getSimpleName();
        evaluation.crossValidateModel(filteredClassifier, instances, foldsNumber, new Random());
        System.out.println(evaluation.toSummaryString(classifierName, true));

        Enumeration<Instance> enumeratedInstances = instances.enumerateInstances();

        Instances filteredInstances = Filter.useFilter(instances, filter);
        while(enumeratedInstances.hasMoreElements()) {
            Instance instance = enumeratedInstances.nextElement();
            System.out.println();
            System.out.println(instance.toString());
            System.out.println(Arrays.toString(filteredClassifier.distributionForInstance(instance)));
        }

        System.out.println(evaluation.toMatrixString("CONFUSION MATRIX"));
    }
}

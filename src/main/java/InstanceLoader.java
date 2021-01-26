import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

public class InstanceLoader {
    private String positiveReviewsLocation = "";
    private String negativeReviewsLocation = "";
    private static String csvFileLocation = "./archive/movie_review.csv";

    public static Instances getInstances() throws IOException {
        File csvFile = new File(csvFileLocation);
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setSource(csvFile);
        return csvLoader.getDataSet();
    }

    public static void printInstances() throws IOException {
        Instances instances = getInstances();
        System.out.println("Number of attributes = " + instances.numAttributes());
        System.out.println("Number of instances = " + instances.numInstances());
        System.out.println("\n\n\t\tInstances\n\n");

        for(int i = 0; i<10; i++) {
            System.out.println("Number of attributes = " + instances.get(i).numAttributes());
        }
    }
}

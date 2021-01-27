import weka.core.*;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static java.util.Arrays.asList;

public class InstanceLoader {
    private static String positiveReviewsLocation = "./archive/movie_reviews/movie_reviews/pos";
    private static String negativeReviewsLocation = "./archive/movie_reviews/movie_reviews/neg";
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

    public static Instances readInstances(long limit) throws IOException {
        File positiveReviewsDirectory = new File(positiveReviewsLocation);
        File negativeReviewsDirectory = new File(negativeReviewsLocation);

        Attribute textAttribute = new Attribute("text", (List<String>) null);
        Attribute labelAttribute = new Attribute("label", asList("pos", "neg"));


        ArrayList<Attribute> attributes = new ArrayList<>(asList(textAttribute, labelAttribute));
        Instances instances = new Instances("instances", attributes, 0);

        for(File positiveFile: Objects.requireNonNull(positiveReviewsDirectory.listFiles())) {
            String review = new BufferedReader(new FileReader(positiveFile)).readLine();
            System.out.println(review);
            DenseInstance instance = new DenseInstance(2 );
            instance.setValue(labelAttribute, "pos");
            instances.add(instance);
        }

        for(File negativeFile: Objects.requireNonNull(negativeReviewsDirectory.listFiles())) {
            String review = new BufferedReader(new FileReader(negativeFile)).readLine();
            DenseInstance instance = new DenseInstance(2 );
            instance.setValue(textAttribute, review);
            instance.setValue(labelAttribute, "neg");
            instances.add(instance);
        }

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }
}

// https://stanfordnlp.github.io/CoreNLP/ner.html


import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;

import java.util.Properties;
import java.util.Set;
import java.util.stream.Collectors;

public class NERPipelineDemo {

    public static void main(String[] args) {
        // set up pipeline properties
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
        // example customizations (these are commented out but you can uncomment them to see the results

        // disable fine grained ner
        // props.setProperty("ner.applyFineGrained", "false");

        // customize fine grained ner
        // props.setProperty("ner.fine.regexner.mapping", "example.rules");
        // props.setProperty("ner.fine.regexner.ignorecase", "true");

        // add additional rules, customize TokensRegexNER annotator
//         props.setProperty("ner.additional.regexner.mapping", "example.rules");
//         props.setProperty("ner.additional.regexner.ignorecase", "true");

        // add 2 additional rules files ; set the first one to be case-insensitive
        // props.setProperty("ner.additional.regexner.mapping", "ignorecase=true,example_one.rules;example_two.rules");

        // set document date to be a specific date (other options are explained in the document date section)
        // props.setProperty("ner.docdate.useFixedDate", "2019-01-01");

        // only run rules based NER
        // props.setProperty("ner.rulesOnly", "true");

        // only run statistical NER
        // props.setProperty("ner.statisticalOnly", "true");

        // set up pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // make an example document
        CoreDocument doc = new CoreDocument("Michael Joseph Jackson, who has $5863, (August 29, 1958 – June 25, 2009) was an American singer, songwriter, and dancer from Rome, Italy. Dubbed the \"King of Pop\", he is regarded as one of the most significant cultural figures of the 20th century. Through stage and video performances, he popularized complicated dance moves such as the moonwalk, to which he gave the name, and the robot. His sound and style have influenced artists of various genres, and his contributions to music, dance, and fashion, along with his publicized personal life, made him a global figure in popular culture for over four decades. Jackson is the most awarded artist in the history of popular music.");
        // annotate the document
        pipeline.annotate(doc);
        // view results
        System.out.println("---");
        System.out.println("entities found");
        for (CoreEntityMention em : doc.entityMentions())
            System.out.println("\tdetected entity: \t"+em.text()+"\t"+em.entityType());
        System.out.println("---");
        System.out.println("tokens and ner tags");

        Set<String> ners = doc.tokens()
                .stream()
                .map(CoreLabel::ner)
                .collect(Collectors.toSet());
        System.out.println("NERS: " + ners.toString());

        String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect(
                Collectors.joining(" "));
        System.out.println(tokensAndNERTags);
    }

}
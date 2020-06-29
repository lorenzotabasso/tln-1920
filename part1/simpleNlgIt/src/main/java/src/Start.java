package src;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.util.ArrayList;

public class Start {

    public static void main(String[] args) {
        // initializing a JSON Parser
        JSONParser jsonParser = new JSONParser();

        // getting the output folder
        File dir = new File("/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part1/output/");
        File[] directoryListing = dir.listFiles();

        ArrayList<String> sentences = new ArrayList<>();

        // For each file (sentence plan) in the output folder, run the Realizer
        if (directoryListing != null) {
            for (File child : directoryListing) {
                try {
                    JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(child));
                    TransformTree generator = new TransformTree(jsonObject.toString());
                    String sentence = generator.realizeSentence();
                    System.out.println(sentence);
                    sentences.add(sentence);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            writeSentencesOnFile(sentences);
        }
    }

    /**
     * Given an ArrayList of sentences, it writes a file for each sentence inside the project's output folder.
     * @param sentences the ArrayList containing all the sentences to print.
     */
    public static void writeSentencesOnFile(ArrayList<String> sentences) {
        String path;
        for (int i = 0; i < sentences.size(); i++) {
            path = "translation" + i + ".txt";
            BufferedWriter writer;
            try {
                writer = new BufferedWriter(new FileWriter("src/main/output/" + path));
                writer.write(sentences.get(i));
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}

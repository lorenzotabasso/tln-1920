package src;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.util.ArrayList;

public class Start {

    public static void main(String[] args) {
        JSONParser jsonParser = new JSONParser();
        File dir = new File("/Users/lorenzotabasso/Desktop/University/TLN/Progetto/19-20/tln-1920/part1/output/");
        File[] directoryListing = dir.listFiles();
        ArrayList<String> sentences = new ArrayList<>();
        if (directoryListing != null) {
            for (File child : directoryListing) {
                try {
                    JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(child));
                    TransformTree generator = new TransformTree(jsonObject.toString());
                    String sentence = generator.getRealisedSentence();
                    System.out.println(sentence);
                    sentences.add(sentence);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            writeSentencesOnFile(sentences);
        }
    }

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

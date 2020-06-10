package src;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import simplenlg.features.Feature;
import simplenlg.features.Gender;
import simplenlg.features.LexicalFeature;
import simplenlg.features.NumberAgreement;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.lexicon.italian.ITXMLLexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.Realiser;

import java.io.IOException;
import java.util.HashMap;

public class TransformTree {

    private final HashMap<Integer, HashMap<String, String>> sentenceTree = new HashMap<>();

    /* We need three different HashMap because each constituent has his own structures and proprieties */
    private final HashMap<Integer, NPPhraseSpec> NPSubTree = new HashMap<>();
    private final HashMap<Integer, PPPhraseSpec> PPSubTree = new HashMap<>();
    private final HashMap<Integer, VPPhraseSpec> VPSubTree = new HashMap<>();
    private final Lexicon italianLexicon = new ITXMLLexicon();
    private final NLGFactory italianFactory = new NLGFactory(italianLexicon);

    public TransformTree(String sentence) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            JsonNode root = objectMapper.readTree(sentence);

            // the second node is null because we are parsing from the root, which has not any parent
            parseJSON(root, null);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Main method of TransformTree. It builds the sentence using a SimpleNLG-It Realizer.
     * @return The realized sentence.
     */
    public String realizeSentence() {
        SPhraseSpec clauseIt = italianFactory.createClause(); // creating the main clause for the sentence

        /* Filling all the NP, PP and VP SubTrees, starting from the 2nd node of sentenceTree, the one who contains
        the first item, until the end of the sentence. */
        fillSubTrees(clauseIt, 2);

        String features = sentenceTree.get(1).get("features");
        if (parseFeatures(features).get("tense").equals("ger")) { // 3rd and 1st sentence
            clauseIt.setFeature(Feature.PROGRESSIVE, true);
            clauseIt.setFeature(Feature.PERFECT, false);
        }

        Realiser realiser = new Realiser();
        return realiser.realiseSentence(clauseIt);
    }

    /**
     * It parse the JSON Sentence Plan and it puts its contents into the {@code sentenceTree} HashMap.
     *
     * @param jsonNode the node from which start the parsing.
     * @param jsonNodeParent the parent of the {@code jsonNode} (if exist. Otherwise it can be null).
     */
    private void parseJSON(JsonNode jsonNode, JsonNode jsonNodeParent) {
        HashMap<String, String> nodeHash = new HashMap<>();
        nodeHash.put("parent", (jsonNodeParent != null) ? jsonNodeParent.get("a").asText() : "null");
        nodeHash.put("type", jsonNode.get("b").asText());
        if (jsonNode.get("c") != null) {
            nodeHash.put("label", jsonNode.get("c").asText());
        }

        if (jsonNode.get("d") != null) {
            nodeHash.put("features", jsonNode.get("d").toString());
        }

        sentenceTree.put(jsonNode.get("a").asInt(), nodeHash);

        ArrayNode arrayNode = (ArrayNode) jsonNode.get("children");
        if (arrayNode == null) {
            return;
        }
        for (int i = 0; i < arrayNode.size(); i++) {
            parseJSON(arrayNode.get(i), jsonNode);
        }
    }

    /**
     * Recursive method to fill all the subtrees (NP, PP and VP).
     * @param startClause the clause of the recursion (normally the root clause).
     * @param key the staring key of the SentenceTree, from which will start the recursion.
     */
    private void fillSubTrees(SPhraseSpec startClause, int key) {
        if (!sentenceTree.containsKey(key)) {
            return; // end of recursion, the sentenceTree is totally filled.
        }

        // the staring node of the recursion.
        HashMap<String, String> node = sentenceTree.get(key);

        if (!node.containsKey("label")) {
            /* If the node not contains the "label" field it means it is a wrapper of internals node of my Sentence
            * Plan. We map the node type to his specific constituent (NPSubTree if it is a NP wrapper,
            * VPSubTree if it is a VP wrapper and PPSubTree if it is a PP wrapper).
            *
            * Simplified assumption: in this if branch, each node will always attach to his startClause by default.
            *
            * Three example to clarify the cases:
            *
            * 1. 3rd sentence, the node with type "subj" will contain all the subj children, so it is a wrapper.
            * 2. 2nd sentence, the node with type "ppcompl" will contain some children, so it is a wrapper.
            * */
            switch (node.get("type")) {
                case "obj": {
                    NPPhraseSpec np = italianFactory.createNounPhrase();
                    NPSubTree.put(key, np);
                    startClause.setObject(np); // Simplified assumption: attach to startClause
                    break;
                }
                case "subj": {
                    NPPhraseSpec np = italianFactory.createNounPhrase();
                    NPSubTree.put(key, np);
                    startClause.setSubject(np); // Simplified assumption: attach to startClause
                    break;
                }
                case "complement":
                    PPPhraseSpec pp = italianFactory.createPrepositionPhrase();
                    PPSubTree.put(key, pp);
                    startClause.addComplement(pp); // Simplified assumption: attach to startClause
                    break;
                case "ppcompl": {
                    NPPhraseSpec np = italianFactory.createNounPhrase();
                    NPSubTree.put(key, np); // "ppcompl" structure is a subtype of NP (for example: "my head")
                    // Attention: in this particular case, its parent will be a "PP", because "ppcompl" is a nested
                    // complement inside a "compl"
                    getPPParent(node).setComplement(np);
                    break;
                }
                case "verb":
                    VPPhraseSpec vp = italianFactory.createVerbPhrase();
                    VPSubTree.put(key, vp);
                    startClause.setVerb(vp); // Simplified assumption: attach to startClause
                    break;
            }
        } else {
            /* In this case, we have to add the children to the respective parent (for example the "prep" object will
            * be added to the corresponding PP Parent). Pay attention to the first 2 cases, which will attach directly
            * to the startClause (simplest hypothesis).
            *
            * Example of first two cases:
            *
            * 1. 1st sentence, all the nodes with types "subj" and "verb" will contain both fields "type" and "label",
            * so they will attach directly to the startClause.
            *
            * 2. 2nd sentence, the node with type "verb" will contain both fields "type" and "label", so it will attach
            * directly to the startClause.
            * */
            switch (node.get("type")) {
                case "subj":
                    startClause.setSubject(node.get("label"));
                    break;
                case "verb":
                    startClause.setVerb(node.get("label"));
                    break;
                case "spec":
                    getNPParent(node).setSpecifier(node.get("label")); // spec is a part of NP
                    break;
                case "noum":
                    getNPParent(node).setNoun(node.get("label")); // noum is a part of NP
                    String features = node.get("features");

                    if (features != null) {
                        String number = parseFeatures(features).get("number");
                        String genre = parseFeatures(features).get("gen");
                        if (number != null) {
                            if (number.equals("pl")) {
                                getNPParent(node).setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
                            } else {
                                getNPParent(node).setFeature(Feature.NUMBER, NumberAgreement.SINGULAR);
                            }
                        }
                        if (genre != null) {
                            // Number
                            if (genre.equals("pl")) {
                                getNPParent(node).setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
                            } else {
                                getNPParent(node).setFeature(Feature.NUMBER, NumberAgreement.SINGULAR);
                            }

                            // Gender
                            if (genre.equals("f")) {
                                getNPParent(node).setFeature(LexicalFeature.GENDER,  Gender.FEMININE);
                            } else {
                                getNPParent(node).setFeature(LexicalFeature.GENDER,  Gender.MASCULINE);
                            }
                        }
                    }
                    break;
                case "prep":
                    getPPParent(node).setPreposition(node.get("label")); // prep is a part of PP
                    break;
                case "ppcompl":
                    getPPParent(node).setComplement(node.get("label")); // ppcompl is a part of PP
                    break;
                case "modifier":
                    getNPParent(node).addModifier(node.get("label")); // modifier is a part of NP
                    break;
                case "vrb":
                    getVPParent(node).setVerb(node.get("label")); // vrb is a part of VP
                    break;
                case "adv":
                    getVPParent(node).addComplement(node.get("label")); // adv is a part of VP
                    break;
            }
        }

        fillSubTrees(startClause, ++key);
    }

    /**
     * Auxiliary method of fillSubTrees. Given a node in the main HashMap, it returns his NP parent.
     *
     * @param node the node in the main HashMap.
     * @return the NP parent of the given node.
     */
    private NPPhraseSpec getNPParent(HashMap<String, String> node) {
        int parent = Integer.parseInt(node.get("parent"));
        return NPSubTree.get(parent);
    }

    /**
     * Auxiliary method of fillSubTrees. Given a node in the main HashMap, it returns his VP parent.
     *
     * @param node the node in the main HashMap.
     * @return the VP parent of the given node.
     */
    private VPPhraseSpec getVPParent(HashMap<String, String> node) {
        int parent = Integer.parseInt(node.get("parent"));
        return VPSubTree.get(parent);
    }

    /**
     * Auxiliary method of fillSubTrees. Given a node in the main HashMap, it returns his PP parent.
     *
     * @param node the node in the main HashMap.
     * @return the PP parent of the given node.
     */
    private PPPhraseSpec getPPParent(HashMap<String, String> node) {
        int parent = Integer.parseInt(node.get("parent"));
        return PPSubTree.get(parent);
    }

    /**
     * It parse and then converts into an HashMap a given string containing a feature (a String associated with the
     * dictionary in part1/dictionary/)
     *
     * @param features String associated with the dictionary in part1/dictionary/
     * @return the HashMap containing the parsed feature
     */
    private HashMap<String, String> parseFeatures(String features) {
        HashMap<String, String> map = new HashMap<>();

        features = features.replace("\"", "");
        features = features.replace("'", "");
        features = features.replace("{", "");
        features = features.replace("}", "");
        String[] splits = features.split(",");
        for (String tuple : splits) {
            String[] parsed = tuple.split(":");
            map.put(parsed[0], parsed[1].trim());
        }

        return map;
    }
}


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class Main {
    private static final String DEFAULT_CONFIG_NAME = "config.txt";
    private static final int N_ARGS_CONFIG = 2;
    private static final String CONFIG_DIR = "configs/";
    private static final String WEIGHTS_DIR = "weights/";
    private static final String INPUT_EXT = ".in";

    public static double[][][] loadData(String dir) throws IOException {
        File f = new File(dir);
        String[] allFiles;
        String[] classes;

        double[][] inputs;
        double[][] outputs;

        allFiles = Arrays.asList(f.listFiles(new FileFilter() {
            public boolean accept(File file) {
                return file.getName().endsWith(INPUT_EXT);
            }
        })).stream().map(e -> (String) e.toString().split("/")[1]).toArray(e -> new String[e]);

        classes = Arrays.asList(allFiles).stream().map(e -> (String) e.substring(0, 1)).distinct().toArray(e -> new String[e]);

        inputs = new double[allFiles.length][];
        outputs = new double[allFiles.length][];

        for (int i = 0; i < allFiles.length; i++) {
            inputs[i] = Arrays.stream(Files.readString(Paths.get(dir + allFiles[i]), StandardCharsets.US_ASCII).split("\n")).mapToDouble(Double::parseDouble).toArray();
            outputs[i] = new double[classes.length];
            outputs[i][Integer.parseInt(allFiles[i].substring(0, 1)) - 1] = 1.0;
        }

        return new double[][][]{inputs, outputs};
    }

    public static Map<String, String> loadConfig(String configName) throws IOException {
        String[] configElems = Files.readString(Paths.get(CONFIG_DIR + configName), StandardCharsets.US_ASCII).split("\n");
        Map<String, String> config = new HashMap<String, String>();
        String[] curElem;

        for (String e : configElems) {
            if(!e.trim().equals("")) {
                curElem = e.split(" ");
                config.put(curElem[0], curElem[1]);
            }
        }

        return config;
    }

    public static double[][][] loadWeights(String fileName, int[] networkShape) throws IOException {
        double [][][] weights = new double[networkShape.length - 1][][];
        BufferedReader weightsFile = new BufferedReader(new FileReader(WEIGHTS_DIR + fileName));

        for(int layer = 0; layer < networkShape.length - 1; layer++) {
            weights[layer] = new double[networkShape[layer]][];
            for(int i = 0; i < networkShape[layer]; i++) {
                weights[layer][i] = new double[networkShape[layer + 1]];
                for(int j = 0; j < networkShape[layer + 1]; j++) {
                    weights[layer][i][j] = Double.parseDouble(weightsFile.readLine());
                }
            }
        }

        weightsFile.close();

        return weights;

    }

    private static void saveWeights(double[][][] weights, String fileName, int[] networkShape) throws IOException {
        BufferedWriter weightsFile = new BufferedWriter(new FileWriter(WEIGHTS_DIR + fileName));

        for(int layer = 0; layer < networkShape.length - 1; layer++) {
            for(int i = 0; i < networkShape[layer]; i++) {
                for(int j = 0; j < networkShape[layer + 1]; j++) {
                    weightsFile.write(weights[layer][i][j] + "\n");
                }
            }
        }

        weightsFile.close();

        return;
    }

    public static void main(String[] args) throws Exception {
        String configName = args.length == N_ARGS_CONFIG ? args[1] : DEFAULT_CONFIG_NAME;
        Map<String, String> config = loadConfig(configName);

        Network network;
        double[][][] weights;
        double[][][] trainingData;
        double[][][] testingData;

        int nInputs = Integer.parseInt(config.get("shape_nInputs"));
        int nHidden1 = Integer.parseInt(config.get("shape_nHidden1"));
        int nHidden2 = Integer.parseInt(config.get("shape_nHidden2"));
        int nOutputs = Integer.parseInt(config.get("shape_nOutputs"));
        double[] randomRange = new double[]{Double.parseDouble(config.get("randomMin")), Double.parseDouble(config.get("randomMax"))};
        System.out.println(Arrays.toString(randomRange));

        int[] networkShape = new int[]{nInputs, nHidden1, nHidden2, nOutputs};

        // prevent running the network without loading in weights
        if(!Boolean.parseBoolean(config.get("weights_loadFromFile")) && !Boolean.parseBoolean(config.get("trainNetwork"))) {
            throw new Exception("Mismatch in the configuration file \"" + configName + "\" -- can't run network without loading in weights.");
        }

        // load weights
        if(Boolean.parseBoolean(config.get("weights_loadFromFile"))) {
            weights = loadWeights(config.get("weights_fileName"), networkShape);
        }
        else {
            weights = null;
        }

        // load training data
        System.out.println("Loading Training Data...");
        trainingData = loadData(config.get("training_data_dirName"));
        

        System.out.println("Building Network..."); 
        
        // create network
        network = new Network(nInputs, nHidden1, nHidden2, nOutputs, randomRange, Boolean.parseBoolean(config.get("trainNetwork")), weights);



        // training vs running network
        if (Boolean.parseBoolean(config.get("trainNetwork"))) {
            System.out.println("Training Network...");
            weights = network.train(trainingData[0], trainingData[1], Integer.parseInt(config.get("training_params_maxIterations")), Double.parseDouble(config.get("training_params_errorThreshold")), Double.parseDouble(config.get("training_params_learningRate")), Double.parseDouble(config.get("training_params_momentum")));
            if(Boolean.parseBoolean(config.get("training_weights_saveToFile"))) {
                saveWeights(weights, config.get("training_weights_fileName"), networkShape);
            }
        }
        
        testingData = loadData(config.get("testing_data_dirName"));

        network.runOverTestingData(testingData[0], testingData[1]);

        System.out.println("\nConfig File: " + configName);
    }
}

import java.text.DecimalFormat;
import java.util.Arrays;

public class Network {
    private static final int N_LAYERS = 3;
    private static final double GET_ERROR_MULT = 0.5;
    private static final double MS_IN_S = 1000;

    private int nInputs;
    private int nHidden1;
    private int nHidden2;
    private int nOutputs;
    private int[] layerSpec;

    private double[] randRange;
    private boolean preloadedWeights;

    private double[] am;

    private double[] omegak;
    private double[] omegaj;

    private double[] psik;
    private double[] psij;
    private double[] psii;

    private double[] thetak;
    private double[] thetaj;
    private double[] thetai;

    private double[] ak;
    private double[] aj;
    private double[] ai;
    
    private double[] Ti;

    private double Esum;

    public double[][] inputSet;
    public double[][] outputSet;
    private int trainingPos;

    private double[][][] weights;

    public Network(int nInputs, int nHidden1, int nHidden2, int nOutputs, double[] randRange, boolean training, double[][][] weights) throws Exception {
        this.nInputs = nInputs;
        this.nHidden1 = nHidden1;
        this.nHidden2 = nHidden2;
        this.nOutputs = nOutputs;

        layerSpec = new int[]{nInputs, nHidden1, nHidden2, nOutputs};

        this.randRange = randRange;
        preloadedWeights = weights != null;

        this.am = new double[nInputs];

        if (training) {
            omegak = new double[nHidden1];
            omegaj = new double[nHidden2];

            psik = new double[nHidden1];
            psij = new double[nHidden2];
            psii = new double[nOutputs];

            thetak = new double[nHidden1];
            thetaj = new double[nHidden2];
            thetai = new double[nOutputs];
        }

        ak = new double[nHidden1];
        aj = new double[nHidden2];
        ai = new double[nOutputs];

        Ti = new double[nOutputs];

        Esum = 0.0;

        trainingPos = -1;

        if (preloadedWeights) {
            if (!verifyWeights(weights)) {
                throw new Exception("The dimensions of the weights file does not match the network dimensions.");
            }
            else {
                this.weights = weights;
            }
        }
        else {
            this.weights = initRandomWeights(randRange[0], randRange[1]);
        }
    }
    
    /**
     * Verifies the dimensions of the given weights object.
     *
     * weights specifies the weights array of which to verify the dimensions.
     *
     * Returns True if weights match the network's layerSpec in dimensions; otherwise
     * returns False.
     */
    private boolean verifyWeights(double[][][] weights) {
        boolean matches = true;

        matches = matches && weights.length == layerSpec.length - 1;

        for (int i = 0; i < layerSpec.length - 1; i++) {
            matches = matches && weights[i].length == layerSpec[i];
            matches = matches && weights[i][0].length == layerSpec[i + 1];
        }
        
        return matches;
    }

    
   /**
    * Initializes the weight array as a 3D array:
    * - D1: the layer
    * - D2: the input node of the next layer
    * - D3: the output node in the previous layer
    * 
    * The array is initialized with random values between randMin (inclusive) and 
    * randMax (inclusive).
    *
    * randMin specifies the minimum random value of the randomly generated values.
    * randMax specifies the maximum random value of the randomly generated values.
    * 
    * Returns the weights array.
    */
    private double[][][] initRandomWeights(double randMin, double randMax) {
        weights = new double[N_LAYERS][][];

        for(int layer = 0; layer < N_LAYERS; layer++) {
            weights[layer] = new double[layerSpec[layer]][];
            for(int i = 0; i < layerSpec[layer]; i++) {
                weights[layer][i] = new double[layerSpec[layer + 1]];
                for(int j = 0; j < layerSpec[layer + 1]; j++) {
                    weights[layer][i][j] = getRandomValue(randMin, randMax);
                }
            }
        }

        return weights;
    }

    /**
     * The output function of each node in the network; in this case, a sigmoid is used.
     *
     * Returns the value of the output function at x.
     */
    private double f(double x) {
        // double e2x = Math.exp(2 * x);
        // return (e2x - 1) / (e2x + 1);
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * The derivative of the output function of each node in the network.
     * 
     */
    private double fDeriv(double x) {
        // double e2x = Math.exp(2 * x);
        // return (4 * e2x) / ((e2x + 1) * (e2x + 1));
        double fx = f(x);
        return fx * (1.0 - fx);
    }

    /**
     * Propogate inputs through network by running computeLayer twice. Only used when training.
     */
    private void runTraining() {
        double omegai = 0.0;

        Esum = 0.0;

        for (int i = 0; i < nOutputs; i++) {

            thetai[i] = 0.0;
            for (int j = 0; j < nHidden2; j++) {

                thetaj[j] = 0.0;
                for (int k = 0; k < nHidden1; k++) {

                    thetak[k] = 0.0;
                    for (int m = 0; m < nInputs; m++) {
                        thetak[k] += am[m] * weights[0][m][k];
                    }

                    ak[k] = f(thetak[k]);
                    thetaj[j] += ak[k] * weights[1][k][j];
                }

                aj[j] = f(thetaj[j]);
                thetai[i] += aj[j] * weights[2][j][i];
            }

            ai[i] = f(thetai[i]);
            omegai = Ti[i] - ai[i];
            psii[i] = omegai * fDeriv(thetai[i]);

            Esum += omegai * omegai;
        }
    
        return;
    }

    /**
     * Propogate inputs through network by running computeLayer twice. Only used when running the network.
     */
    private void run() {
        double omegai, thetai, thetaj, thetak = 0.0;

        Esum = 0.0;

        for (int i = 0; i < nOutputs; i++) {
            thetai = 0.0;

            for (int j = 0; j < nHidden2; j++) {
                thetaj = 0.0;

                for (int k = 0; k < nHidden1; k++) {

                    thetak = 0.0;
                    for (int m = 0; m < nInputs; m++) {
                        thetak += am[m] * weights[0][m][k];
                    }

                    ak[k] = f(thetak);
                    thetaj += ak[k] * weights[1][k][j];
                }

                aj[j] = f(thetaj);
                thetai += aj[j] * weights[2][j][i];
            }

            ai[i] = f(thetai);
            omegai = Ti[i] - ai[i];

            Esum += omegai * omegai;
        }

        return;
    }

    /**
     * Returns a random value between randMin (inclusive) and randMax (inclusive) to each element.
     *
     * randMin specifies the minimum random value
     * randMax specifies the maximum random value
     */
    private double getRandomValue(double randMin, double randMax) {
        return Math.random() * (randMax - randMin) + randMin;
    }

    private void printNetworkSpecs() {
        System.out.println("\nNumber of Inputs: " +  nInputs);
        System.out.println("Number of Nodes in Hidden Layer 1: " + nHidden1);
        System.out.println("Number of Nodes in Hidden Layer 2: " + nHidden2);
        System.out.println("Number of Outputs: " + nOutputs);
        System.out.println("Random Value Range: " + Arrays.toString(randRange));

        return;
    }

    /**
     * Returns the error of the network.
     */
    private double getError() {
        return GET_ERROR_MULT * Esum;
    }

    public double[][][] train(double[][] inputs, double[][] outputs, int maxIterations, double errorThreshold, double lr, double momentum) {
        int iterations;
        double totalError;
        int trainingLen;
        double trainingTime;
        double[][] curTrainingMember;
        double[][][] deltas;

        boolean finished, errorThresholdReached, maxIterationsReached;

        finished = false;
        errorThresholdReached = false;
        maxIterationsReached = false;

        iterations = 0;
        totalError = 0.0;
        trainingLen = inputs.length;
        trainingTime = System.currentTimeMillis();

        inputSet = inputs;
        outputSet = outputs;

        deltas = new double[N_LAYERS][][];
        for(int layer = 0; layer < N_LAYERS; layer++) {
            deltas[layer] = new double[layerSpec[layer]][];
            for(int i = 0; i < layerSpec[layer]; i++) {
                deltas[layer][i] = new double[layerSpec[layer + 1]];
                for(int j = 0; j < layerSpec[layer + 1]; j++) {
                    deltas[layer][i][j] = 0.0;
                }
            }
        }

        while (!finished) {
            curTrainingMember = getNextTrainingMember();
            am = curTrainingMember[0];
            Ti = curTrainingMember[1];

            runTraining();

            for(int j = 0; j < nHidden2; j++) {
                omegaj[j] = 0.0;
                for(int i = 0; i < nOutputs; i++) {
                    omegaj[j] += psii[i] * weights[2][j][i];
                    deltas[2][j][i] = lr * aj[j] * psii[i] + momentum * deltas[2][j][i];
                    weights[2][j][i] += deltas[2][j][i];
                }

                psij[j] = omegaj[j] * fDeriv(thetaj[j]);
            }

            for(int k = 0; k < nHidden1; k++) {
                omegak[k] = 0.0;
                for(int j = 0; j < nHidden2; j++) {
                    omegak[k] += psij[j] * weights[1][k][j];
                    deltas[1][k][j] = lr * ak[k] * psij[j] + momentum * deltas[1][k][j];
                    weights[1][k][j] += deltas[1][k][j];
                }

                psik[k] = omegak[k] * fDeriv(thetak[k]);

                for(int m = 0; m < nInputs; m++) {
                    deltas[0][m][k] = lr * am[m] * psik[k] + momentum * deltas[0][m][k];
                    weights[0][m][k] += deltas[0][m][k];
                }
            }

            iterations++;
            totalError += getError();
            if (trainingPos == trainingLen - 1) {
                System.out.println(iterations + " - " + totalError);
                errorThresholdReached = totalError <= errorThreshold;
                totalError = 0.0;
            }

            maxIterationsReached = iterations >= maxIterations;

            finished = maxIterationsReached || errorThresholdReached;
        }

        trainingTime = System.currentTimeMillis() - trainingTime;

        if (errorThresholdReached) {
            System.out.println("Network has reached the error threshold.");
        }
        if (maxIterationsReached) {
            System.out.println("Network has reached the maximum number of iterations.");
        }

        System.out.println("\nMax Iterations: " + maxIterations);
        System.out.println("Error Threshold: " + errorThreshold);
        System.out.println("Learning Rate: " + lr);
        System.out.println("Use Preloaded Weights: " + preloadedWeights);

        printNetworkSpecs();

        System.out.println("\nTraining Time: " + (trainingTime / MS_IN_S) + " seconds");

        
        return weights;
    }

    private double[][] getNextTrainingMember() {
        trainingPos += 1;

        if (trainingPos == inputSet.length) {
            trainingPos = 0;
        }

        return new double[][]{inputSet[trainingPos], outputSet[trainingPos]};
    }

    // TODO: write this
    public void runOverTestingData(double[][] inputs, double[][] outputs) {
        int networkIdx = 0;
        int trueIdx = 0;
        DecimalFormat df = new DecimalFormat("0.00");

        for (int i = 0; i < inputs.length; i++) {
            am = inputs[i];
            Ti = outputs[i];

            run();

            trueIdx = 0;
            for (int j = 0; j < Ti.length; j++) {
                trueIdx = Ti[j] > Ti[trueIdx] ? j : trueIdx;
            }            

            networkIdx = 0;
            for (int j = 0; j < ai.length; j++) {
                networkIdx = ai[j] > ai[networkIdx] ? j : networkIdx;
            }

            System.out.println("\n\nnetwork's prediction for " + (trueIdx + 1)  + ": " + (networkIdx + 1));
            Arrays.stream(ai).forEach(e -> System.out.print(df.format(e) + " " ));
        }
    }
}

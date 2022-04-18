package dn2;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Encapsulates a DN and the functionality around it.
 */
public class DNHandler {

    private static boolean initialized = false;

    //boolean for tracking if DN has been loaded, fixed or must be initialized
    public static boolean loadedDN = false;

//    //boolean to indicate RenderScript DN
//    public static Boolean renderscript = false;

    //boolean to indicate frozen DN
    public static Boolean frozen = false;

    // Application context
//    private Context mContext;

    //initialize current input (X) as well as current and new motor (Z)
    private static ArrayList<float[][]> input;
    private static ArrayList<float[][]> currentMotor;
    private static ArrayList<float[][]> newMotor;

    private static int numNeuronsPassed; //variable for number of neurons per hidden layer passed by user to class
    private static int[] numNeurons; // number of hidden neurons for each hidden layer (uses numNeuronsPassed to initialize)

    private static int numSensor; //number of sensors passed based off 3D array passed (dynamic)
    private static int[] sensorHeight; //stores height of each sensor based off 3D array passed (dynamic)
    private static int[] sensorWidth; //stores width of each sensor based off 3D array passed (dynamic)

    private static int numMotor; //number of motors passed based off 3D array passed (dynamic)
    private static int[] motorHeight; //stores height of each motor based off 3D array passed (dynamic)
    private static int[] motorWidth; //stores width of each motor based off 3D array passed (dynamic)

    private static int[][] inputSize; // The dimensions of the input passed stored in 2D array at the moment for rf and other dynamic setting {height, width)
    private static int[][] motorSize;  // The dimensions of the motor passed from the environment in 2D array at the moment for rf and other dynamic setting {height, width)

    private static int numHidden = 1; //Number of hidden layers // default 1 hidden layer

    private static int YtopkVal = 1; //Value to be used in YTopK initialization // default 1
    //private static int ZtopkVal = 1; //Value to be used in ZTopK initialization // default 1
    private static int maxYTopK; // The number of top-k neurons per hidden layer //default set to 1
    private static int[] Ztopk; // The number of top-k neurons per motor layer //default setprivate
    private static float pre_screen = .5f; // The percentage of filtering to boost the pattern matching part of the top-k competition //default .5f

    private static float dopamineMultiplier = .0f;
    private static float serotoninMultiplier = .0f;

    private static int[][] rfSizes; //receptive field size //default to full sensor
    private static int[][] rfStrides; //receptive field stride //default to (1,1)
    private static int[] rfSizesHeight = {0}; //height variable of the rfSize
    private static int[] rfSizesWidth = {0}; //width variable of the rfSize
    private static int[] rfStridesHeight = {0}; //height variable of the rfStride
    private static int[] rfStridesWidth = {0}; //height variable of the rfStride

    //as initializing any of these to zero would make no sense, this is used to check if the defaults should be used, or if the user set their own value

    private static int[][] rf_id_loc; //used to indicate the center of each receptive field with rfSize and rfStride

    private static String growth; // Filename with the growth rate table.
    private static String mean; // Filename with the mean table
    private static String neuronGrowth;
    private static float[][] growthtable; // This table determines how flexible the network creates new neurons depending on the remaining number of available neurons.
    private static float[][] meantable; // This table determines how flexible the network determines a hidden neuron is a winner from the top-k competition depending on the remaining number of available neurons.
    private static float[][] neuronGrowthTable;

    private static int[] neuronTypes = new int[] {0,0,0,0,2,0,0}; // Number of initial neurons for each of the neuron types: 001, 010, 011, 100, 101, 110, 111 //2 type 5 by default

    public static int[] computeResponseWeights = new int[]{1,1,1}; //response weights used in computeResponse for bottomUpResponse, lateralResponse, and topDown
    public static int[] computeMotorResponseWeights = new int[]{1,1}; //response weights used in computeMotorResponse for bottomUpResponse, and lateralResponse

    private static int where_count;

    // Decides which process to use when calculating responses
    public enum RESPONSE_CALCULATION_TYPE {GOODNESS_OF_MATCH, TRIANGLE_RESPONSE}
    // Response type for the hidden layer. Default: Goodness of match.
    private RESPONSE_CALCULATION_TYPE hiddenResponseType = RESPONSE_CALCULATION_TYPE.GOODNESS_OF_MATCH;
    // Response type for the motor layer. Default: Goodness of match.
    private RESPONSE_CALCULATION_TYPE[] motorResponseType = {RESPONSE_CALCULATION_TYPE.GOODNESS_OF_MATCH};

    private static float[][] sweet;
    private static float[][] pain;

    private DN2 network; // DN2 instance used throughout the program


    /**
     * DNHandler constructor requiring minimal user input (all other variables can be changed with setters).
     * Otherwise, the DN's default values are used.
     * @param neuronsPerHiddenLayer Number of hidden neurons per Hidden area.
     * @param growthRateTablePath Path to the growth rate table.
     * @param meanValueTablePath Path to the mean value table.
     * @param context The Android context.
     */
    public DNHandler(int neuronsPerHiddenLayer, String growthRateTablePath, String meanValueTablePath, String neuronGrowthPath) {

        //initialize all data passed
        numNeuronsPassed = neuronsPerHiddenLayer;

//        mContext = context; //sets passed context
        growth = growthRateTablePath; //passes path to String var
        mean = meanValueTablePath; //passes path to String var
        neuronGrowth = neuronGrowthPath;

        growthtable = getTablevector(growth);
        meantable = getTablevector(mean); //gets table using passed string
        neuronGrowthTable = getTablevector(neuronGrowth);

        loadedDN = false; //for tracking that DN is not loaded
    }

    //commented out general initialization due to simplicity and cleanliness of code (can easily be added back upon request or desire)
//    /**
//     * DNHandler constructor requiring specification of all variables required for initializing DN
//     * - to be used by settings file or user who wants to specify all DN logic.
//     * @param neuronsPerHiddenLayer Number of hidden neurons per Hidden area.
//     * @param numberHiddenLayersSet Total number of hidden areas in the network.
//     * @param pre_screenSet Threshold value of acceptable euclidean match of each component.
//     * @param neuronTypesSet Starting number of hidden neurons for each DN2 type.
//     * @param YtopkSet TopK value for each HiddenLayer.
//     * @param ztopk Array of topK value for each motor area.
//     * @param rfSizeSet The size of the receptive field.
//     * @param rfStrideSet The number of shifts the receptive field will do.
//     * @param growthRateTablePath Path to the growth rate table.
//     * @param meanValueTablePath Path to the mean value table.
//     * @param computeResponseWeightsSet The relative weights of bottom-up, lateral, and top-down responses.
//     * @param hardSymbolicSet Whether the motor response is hard symbolic or not. (True: hard symbolic, False: soft symbolic)
//     * @param context The Android context.
//     */
//    public DNHandler(int neuronsPerHiddenLayer, int numberHiddenLayersSet, float pre_screenSet, int[] neuronTypesSet, int YtopkSet, int[] ztopk,
//                     int [] rfSizeSet, int[] rfStrideSet, String growthRateTablePath, String meanValueTablePath, String neuronGrowthPath, int[] computeResponseWeightsSet, int[] computeMotorResponseWeightsSet, boolean[] hardSymbolicSet, Context context) {
//
//        //initialize all data passed
//        numNeuronsPassed = neuronsPerHiddenLayer; //sets numNeuronsPassed given user input
//        numHidden = numberHiddenLayersSet; //sets numHidden given user input
//        pre_screen = pre_screenSet; //sets pre_screen given user input
//        neuronTypes = neuronTypesSet; //sets neuronTypes given user input
//        YtopkVal = YtopkSet; //sets YtopkVal given user input
//        Ztopk = ztopk; //sets ZtopkVal given user input
//        computeResponseWeights = computeResponseWeightsSet; //sets computeResponseWeights given user input
//        computeMotorResponseWeights = computeMotorResponseWeightsSet; //sets computeMotorResponseWeights given user input
//        mContext = context; //sets passed context
//        growth = growthRateTablePath; //passes path to String var
//        mean = meanValueTablePath; //passes path to String var
//        neuronGrowth = neuronGrowthPath;
//        growthtable = getTablevector(growth, mContext); //gets table using passed string
//        meantable = getTablevector(mean, mContext); //gets table using passed string
//        neuronGrowthTable = getTablevector(neuronGrowthPath, mContext);
//
//    }

    //if the user wants to initialize the DN before use due to the DN taking time to initialize, this function can be called after the class is initialized
    //This initializes the DN calling the Update function for firstRun for training
    //For this to not affect DN logic, the user must pass the sensorInitializationVector and motorInitializationVector which they are expected to pass on the first Update call
    //This is required as this class dynamically assigns variable values given the input, making the user specify less, but requiring the motor and sensor input

    /**
     * Initializes the DN.
     * @param sensorInitializationVector A 3-dimensional float array of the first sensory input.
     * @param motorInitializationVector A 3-dimensional float array of the first motor context input.
     */
    public void initializeDN(ArrayList<float[][]> sensorInitializationVector, ArrayList<float[][]> motorInitializationVector) {
        initializeVectors(sensorInitializationVector,motorInitializationVector);
        DNsetup();
        Update(sensorInitializationVector, motorInitializationVector, null, null,true);
    }

    /**
     * Dynamically initializes the sensory input and motor input arrays, as well as receptive field,
     * top k, number of neurons, and variables for DN logic. This function is called in initializeDN
     * or Update depending on how the user chooses to initialize the DN.
     * @param sensorInput A 3-dimensional float array of the sensory input.
     * @param motorInput A 3-dimensional float array of the motor context input.
     */
    private static void initializeVectors(ArrayList<float[][]> sensorInput, ArrayList<float[][]> motorInput){

        //dynamically get sensor information from passed sensor
        numSensor = sensorInput.size(); //set number of sensors by first column of float array
        sensorHeight = new int[numSensor]; //initialize sensorHeight given numSensor
        sensorWidth = new int[numSensor]; //initialize sensorWidth given numSensor
        for (int i = 0; i < numSensor; i++){ //sets sensor heights and widths by input
            sensorHeight[i] = sensorInput.get(i).length;
            sensorWidth[i] = sensorInput.get(i)[0].length;
        }

        //dynamically get motor information from passed motor
        numMotor = motorInput.size(); //set number of motors by first column of float array
        motorHeight = new int[numMotor]; //initialize motorHeight given numMotor
        motorWidth = new int[numMotor]; //initialize motorWidth given numMotor
        for (int i = 0; i < numMotor; i++){ //sets motor heights and widths by input
            motorHeight[i] = motorInput.get(i).length;
            motorWidth[i] = motorInput.get(i)[0].length;
        }

        rfSizes = new int[numSensor][2]; // Allow 2D receptive fields
        rfStrides = new int[numSensor][2]; // allow 2D strides
        for (int i = 0; i < numSensor; i++) {
            if (rfSizesHeight[i] == 0){ //if never set, set height of rfSize to full view, else sets to set value
                rfSizes[i][0] = sensorHeight[i];
            } else {
                rfSizes[i][0] = rfSizesHeight[i];
            }
            if (rfSizesWidth[i] == 0){ //if never set, set width of rfSize to full view, else sets to set value
                rfSizes[i][1] = sensorWidth[i];
            } else {
                rfSizes[i][1] = rfSizesWidth[i];
            }
            if (rfStridesHeight[i] == 0) { //if never set, set height of rfStride to full view, else sets to set value
                rfStrides[i][0] = 1;
            } else {
                rfStrides[i][0] = rfStridesHeight[i];
            }
            if (rfStridesWidth[i] == 0){ //if never set, set width of rfStride to full view, else sets to set value
                rfStrides[i][1] = 1;
            } else {
                rfStrides[i][1] = rfStridesWidth[i];
            }
        }

        //TODO check why not per sensor and only first sensor for rfSize and rfStride
        inputSize = new int[numSensor][2]; //setting inputSize values to be used by rf_id_loc for calibration
        for (int i = 0; i < numSensor; i++){
            inputSize[i][0] = sensorHeight[i];
            inputSize[i][1] = sensorWidth[i];
        }
        motorSize = new int[numMotor][2]; //setting motorSize values to be used by rf_id_loc for calibration
        for (int i = 0; i < numMotor; i++){
            motorSize[i][0] = motorHeight[i];
            motorSize[i][1] = motorWidth[i];
        }

        // Indicates the center of each receptive field with rfSize and rfStride
        rf_id_loc = new int[][]{{0}}; //TODO delete everywhere, as it is never used.

//        Ztopk = new int[numMotor];
//        for (int i = 0; i < numMotor; i++){ //sets top k values for all motors
//            Ztopk[i] = ZtopkVal;
//        }


        numNeurons = new int[numHidden];
        for (int i = 0; i < numHidden; i++){ //sets pre-initialized neurons for all hidden layers
            numNeurons[i] = numNeuronsPassed;
        }

        //initialize variables to be used for DN logic in Update function
        input = new ArrayList<>(); //used for sensor computation
        currentMotor = new ArrayList<>(); //used for motor training
        newMotor = new ArrayList<>(); //used for motor training
        for(int i = 0; i < numMotor; i++) { //initialize to proper size given initialization values
            currentMotor.add(i, new float[motorSize[i][0]][motorSize[i][1]]);
            newMotor.add(i, new float[motorSize[i][0]][motorSize[i][1]]);
        }
        for(int i = 0; i < numSensor; i++){ //initialize to proper size given initialization values
            input.add(new float[inputSize[i][0]][inputSize[i][1]]);
        }

        // TODO: Create reinforcement sensors so it doesn't need to be added manually
        sweet = new float[1][numMotor];
        pain = new float[1][numMotor];

        input.add(pain);
        input.add(sweet);

    }

    /**
     * Sets up and initializes the DN (to be called after initializeVectors).
     */
    private void DNsetup () {

        // Prepare the DN program (defaults set)
        int[] lateral_zone = {};
        int lateral_length = 0;
        float lateral_percent = 0;

//        int numInput, int[][] inputSize, int numMotor, int[][] motorSize, int[] topKMotor,
//        int numHidden, int[] rfSize, int[] rfStride, int[][] rf_id_loc, int[] numHiddenNeurons,
//        int[] topKHidden, float prescreenPercent, int[] typeNum, float[][] growthrate, float[][] meanvalue,
//        float lateralpercent, boolean dynamicInhibition, int[] lateral_zone, int lateral_length, MODE network_mode, int Zfrequency, float[][] neuronGrowth, MotorLayer.RESPONSE_CALCULATION_TYPE motorResponseType, HiddenLayer.RESPONSE_CALCULATION_TYPE hiddenResponseType
        
        network = new DN2(numSensor, inputSize, numMotor, motorSize, Ztopk,
                numHidden, rfSizes, rfStrides, rf_id_loc, numNeurons, pre_screen,
                neuronTypes, growthtable, meantable, lateral_percent, true,
                lateral_zone, lateral_length, DN2.MODE.NONE, neuronGrowthTable, motorResponseType,
                hiddenResponseType
        );

        //set initialized to true
        initialized = true;
    }

    //booleans to monitor if its the first and/or second running for DN calibration purposes
    public static boolean firstRun = true;
    public static boolean secondRun = false;

    /**
     * Loops through a single update of the DN.
     * @param sensorInput A 3-dimensional float array of the sensory input to be given to the DN.
     * @param motorInput A 3-dimensional float array of the motor context input to be given to the DN.
     * @param learn_flag Boolean deciding if DN is learning or frozen. (True: Learning, False: Frozen)
     * @return Returns the motor response of the DN if the DN is not in learning mode.
     *         If learning is true, returns motorInput.
     */
    public ArrayList<float[][]> Update(ArrayList<float[][]> sensorInput, ArrayList<float[][]> motorInput, boolean[] reward, boolean[] punish, boolean learn_flag) {

        input = sensorInput;

        if (reward != null){
            for (int i = 0; i < numMotor; i++){
                if (reward[i]){
                    sweet[0][i] = 1.0f;
                }
            }
        }

        if (punish != null) {
            for (int i = 0; i < numMotor; i++){
                if (punish[i]){
                    pain[0][i] = 1.0f;
                }
            }
        }

        //if its the fist run for training, initial values must be inputed into DN per logic
        if (firstRun) { //&& learn_flag){

            if (!initialized && !loadedDN){ // check if DN has been initialized be the user, if not, does so
                initializeVectors(sensorInput, motorInput);
                DNsetup();
            } else if (!initialized && loadedDN){
                initializeVectors(sensorInput, motorInput);
            }

            //passes input to currentMotor and input, requires for later learning
            currentMotor = motorInput;

            //prepare for second run
            firstRun = false;
            secondRun = true;

            network.setSerotoninMultiplier(serotoninMultiplier);
            network.setDopamineMultiplier(dopamineMultiplier);

            //this was used to debug the pre-responses at one point (not necessary when not debugging preresponses)
//            try {
//                myWriter = new FileWriter(Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "AOS/preResponses.txt", true);
//            } catch (IOException e) {
//                System.out.println("An error occurred.");
//                e.printStackTrace();
//            }

        } else if (secondRun) { //&& learn_flag) {

            //add reinforcers
            input.add(pain);
            input.add(sweet);

            //network.computeHiddenResponse(input, currentMotor, learn_flag);
            //the below is added as learn_flag is always true

            //does first training on DN using initialized input from first run
            network.computeHiddenResponse(input, currentMotor, learn_flag);

            //sets currentMotor to new input for next call
            currentMotor = motorInput;

            //newMotor = network.computeMotorResponse();

            //sets sensorInput to input for next call
            //input = sensorInput;

            //tells the DN to shift and replace the previous motor and hidden responses as well.
            network.replaceMotorResponse();
            network.replaceHiddenResponse();

            //sets second run to false so DN trains normally from now on
            secondRun = false;

        } else {

            //add reinforcers
            input.add(pain);
            input.add(sweet);

            // this was once used to debug pre-responses. To use, comment in all myWriter variables
//            ArrayList<float[]> motorPreResponses;
//            motorPreResponses = network.getMotorPreResponses();
//
////            for (int i = 0; i < numMotor; i++){
//            for (int i = 0; i < 1; i++){
//                if(motorPreResponses.get(i) != null){
//                    for(int j = 0; j < motorPreResponses.get(i).length; j++){
//                        try {
//                            myWriter.append(motorPreResponses.get(i)[j] + " ");
////                            System.out.print(motorPreResponses.get(i)[j] + " ");
//                        } catch (IOException e) {
//                            System.out.println("An error occurred.");
//                            e.printStackTrace();
//                        }
//                    }
//
//                    try {
//                        myWriter.append("\n");
//                        myWriter.flush();
//                        System.out.println();
//                    } catch (IOException e) {
//                        System.out.println("An error occurred.");
//                        e.printStackTrace();
//                    }
//                }
//            }

            //computes DN response for training or testing
            if (!frozen){
                network.computeHiddenResponse(input, currentMotor, true); //always true (always learning)
            } else {
                network.computeHiddenResponse(input, currentMotor, learn_flag);
            }

            //sets currentMotor to new input for next call
            currentMotor = motorInput;

            if (learn_flag) { //if DN is learning (supervising)
                newMotor = network.computeMotorResponse(pain, sweet);
                //sets the newMotor as the currentMotor
                newMotor = currentMotor;
                //updates the supervised weights, training the DN for the last time step given the motor context
                network.updateSupervisedMotorWeights(newMotor);
            } else { //if learning is performing (reinforcement learning)
                //sets newMotor to the computed response for the DN
                newMotor = network.computeMotorResponse(pain, sweet); //Added serotonin and dopamine input
                //assigns the newMotor to the currentMotor for context to next computation
                currentMotor = newMotor;
                //update motorWeights (RECENTLY ADDED)
                if (!frozen){
                    network.updateMotorWeights();
                }
            }

            //tells the DN to shift and replace the previous motor and hidden responses as well.
            network.replaceMotorResponse();
            network.replaceHiddenResponse();
        }

        //returns the newMotor because if in testing mode, this is the DN's computed response
        return newMotor;

    }

    //this function saves a serialized DN and resets the booleans for tracking first or second run if the saved DN is to be trained again
    //Requires the name that the user wants to save the DN with and saves to phones app file directory

    /**
     * Serializes the DN and saves it to a file with the given path/name.
     * @param pathName The path/name of the DN file to be created.
     */
    public void saveDN(String pathName){
        System.out.println("SAVING DN");
//        System.out.println( mContext.getFilesDir());
//        mContext.getFilesDir();
//        if(renderscript)
//        {
//            network.saveAllocations();
//        }
        network.serializeSave(pathName);
        System.out.println("FINISHED SAVING");
        //reset booleans for next training sequence (they are static variables specific to DNHandler class)
        firstRun = true;
        secondRun = false;

        //used for pre-responses debugging (comment in to use)
//        try {
//            myWriter.close();
//        } catch (IOException e) {
//            System.out.println("An error occurred.");
//            e.printStackTrace();
//        }

    }

    public void createDefaultGrowthRateTable()
    {
        growthtable = new float[50][9];
        double percent = 0;
        float topk = 1f / numNeuronsPassed;
        float divider = 1f / maxYTopK;

        for(int i = 0; i < growthtable.length; i++)
        {
            percent += 0.02f;
            growthtable[i][0] = (float)percent;
            for(int j = 1; j < 8; j++)
            {
                growthtable[i][j] = 1.0f;
            }

            growthtable[i][8] = topk;
//            System.out.println(topk);

            if(percent > divider)
            {
                topk += 1f / numNeuronsPassed;
                if (topk > (maxYTopK / 100f)*numNeuronsPassed)
                    topk = (maxYTopK / 100f)*numNeuronsPassed;
                divider += 1f / maxYTopK;
            }
        }
//        System.out.println();
//        System.out.println(Arrays.deepToString(growthtable));
    }

    //this function loads a serialized DN
    //Requires the name that the user saved their DN under and grabs the DN from the phones app file directory

    /**
     * Deserializes a DN file and loads the DN into the DNHandler object.
     * @param pathName The path to the file containing the DN.
     * @throws IOException If file is not found.
     * @throws ClassNotFoundException If class is not found.
     */

    public void loadDN(String pathName) throws IOException, ClassNotFoundException {
        System.out.println("OPEN DN");
//        System.out.println(mContext.getFilesDir());
//        mContext.getFilesDir();
        System.out.println(pathName);
//        network.deserializeLoad(pathName);
//        System.out.println("FINISHED OPENING");

        try{
            FileInputStream fileIn = new FileInputStream(pathName);
            ObjectInputStream in = new ObjectInputStream(fileIn);
//            if(renderscript)
//            {
//                network = (RS_DN2) in.readObject();
//                network.reinitializeRenderScript(mContext);
//                network.loadAllocations();
//
//                network.setNeuronGrowthRate(neuronGrowthTable);
//            }
//            else {
                network = (DN2) in.readObject();

                network.setNeuronGrowthRate(neuronGrowthTable);
//            }
            in.close();
            fileIn.close();
        }catch(IOException i) {
            i.printStackTrace();
        }

        firstRun = true;
        secondRun = false;
        initialized = false;
        loadedDN = true;

        for(int i = 0; i < numMotor; i++) { //re-initialize to zero so old motor response is not outputted
            newMotor.set(i, new float[motorSize[i][0]][motorSize[i][1]]);
        }

        //resetRS();

    }

//    //this function loads a serialized DN over an already loaded DN - it is not efficient and should not be used if possible
//    //Requires the name that the user saved their DN under and grabs the DN from the phones app file directory
//    public void loadDNold(String pathName) throws IOException, ClassNotFoundException {
//        System.out.println("OPEN DN");
//        System.out.println(mContext.getFilesDir());
//        mContext.getFilesDir();
//        System.out.println(pathName);
//        network.deserializeLoad(pathName);
//        System.out.println("FINISHED OPENING");
//        for(int i = 0; i < numMotor; i++) { //re-initialize to zero so old motor response is not outputted
//            newMotor.set(i, new float[motorSize[i][0]][motorSize[i][1]]);
//        }
//    }

//    /**
//     * Sets boolean to indicate if running RenderScript DN or not
//     * @param renderscriptSet Run RenderScript DN when true.
//     */
//    public void setRenderScriptDN(Boolean renderscriptSet) {
//        renderscript = renderscriptSet;
//    }

    /**
     * Sets boolean to indicate if running frozen DN for testing or not
     * @param frozenDNBoolean Run testing with frozen DN when true.
     */
    public void setFrozenDN(Boolean frozenDNBoolean) {
        frozen = frozenDNBoolean;
    }

    /**
     * Sets the number of hidden layers.
     * @param numHiddenSet The number of hidden layers to be set.
     */
    public void setNumHidden(int numHiddenSet) {
        numHidden = numHiddenSet;
    }

    /**
     * Set the pre-screen percent.
     * @param pre_screenSet The pre-screen percent to be set.
     */
    public void setPre_screen(float pre_screenSet) {
        pre_screen = pre_screenSet;
    }

    /**
     * Sets the initial neuron types. The DN will have top-k+1 initial neurons of each type specified
     * by a nonzero value in the corresponding index. Neuron type-1 is represented by index 0 and
     * neuron type-7 is represented by index 6.
     * @param neuronTypesSet The array of neuron types specifying the neuron types the DN will
     *                       start out with.
     */
    public void setNeuronTypes (int[] neuronTypesSet) {
        if (neuronTypesSet.length == 7) {
            neuronTypes = neuronTypesSet;
        } else {
            throw new IllegalArgumentException("neuronTypesSet format does not match required format of {n, n, n, n, n, n, n}");
        }
    }

    /**
     * Sets the top-k value for the hidden (Y) area.
     * @param YtopkSet The top-k value to be set.
     */
    public void setMaxYtopk (int YtopkSet) {
        maxYTopK = YtopkSet;
    }

//    /**
//     * Sets the top-k value for the motor (Z) area.
//     * @param ZtopkSet The top-k value to be set.
//     */
//    public void setZtopk(int ZtopkSet) {
//        ZtopkVal = ZtopkSet;
//    }

    public void setZtopk(int[] ztopkvalues)
    {
        Ztopk = ztopkvalues;
    }

    // TODO: Check that comment is correct.
    /**
     * Sets the receptive field sizes.
     * @param rfHeightSizeSet The height of each Y neuron's receptive field.
     * @param rfWidthSizeSet The width of each Y neuron's receptive field.
     */
    public void setRfSizes (int[] rfHeightSizeSet, int[] rfWidthSizeSet){
        rfSizesHeight = rfHeightSizeSet;
        rfSizesWidth = rfWidthSizeSet;
    }

    // TODO: Check that comment is correct.
    /**
     * Set the receptive field stride.
     * @param rfHeightStrideSet The column shift for each receptive field.
     * @param rfWidthStrideSet The row shift for each receptive field.
     */
    public void setRfStrides (int[] rfHeightStrideSet, int[] rfWidthStrideSet){
        rfStridesHeight = rfHeightStrideSet;
        rfStridesWidth = rfWidthStrideSet;
    }

    /**
     * Sets the weights for DN compute response in the hidden layer.
     * @param computeResponseWeightsSet The weights to be set for DN compute response.
     */
    public void setComputeResponseWeights (int[] computeResponseWeightsSet) {
        computeResponseWeights = computeResponseWeightsSet;
    }

    /**
     * Sets the weights for DN compute response in the motor layer.
     * @param motorComputeResponseWeightsSet The weights to be set for DN compute response.
     */
    public void setMotorComputeResponseWeights (int[] motorComputeResponseWeightsSet) {
        computeMotorResponseWeights = motorComputeResponseWeightsSet;
    }

    /**
     * Sets the weights for dopamine and serotonin.
     * @param dopSerWeights The vales of dopamine and serotinin multipliers
     */
    public void setDopSerWeights (float[] dopSerWeights) {
        dopamineMultiplier = dopSerWeights[0];
        serotoninMultiplier = dopSerWeights[1];
    }

    // Read a file and return a 2D array of floats - used for getting tables for growth table of neurons and mean value table

    /**
     * Reads a file and returns a 2D array of floats - used for getting tables such as the growth table
     * of neurons and the mean value table.
     * @param filename The name/path of the file containing the table to be read.
     * @param context The android context.
     * @return null
     */
    private static float[][] getTablevector(String filename){
        try {
            return new TableReader(filename).getTable();
        } catch (IOException e) {
            System.err.println("Issues while opening the table");
            e.printStackTrace();
        }

        return null;
    }


    // TODO: Figure out what this does and add a comment!
    /**
     * Configures where count.
     * @param rfSize Size of the receptive field.
     * @param rfStride Stride of the receptive field.
     * @param inputSize Size of input.
     * @return Receptive field ID location.
     */
    private static int[][] configure_where_count(int[] rfSize, int[] rfStride,
                                                 int[][] inputSize) {
        // in matlab we use the rf_id in col major order.
        // but rf_id_loc is [id][height][width]
        where_count = 0;
        int[][] rfIdLoc;
        if(rfSize[0]!=0 && rfSize[1]!=0){
            int half_rf_size_w = (rfSize[1] - 1) / 2;
            int half_rf_size_h = (rfSize[0] - 1) / 2;

            //System.out.println(half_rf_size_w + " " + half_rf_size_h);
            //System.out.println(Integer.toString(inputSize[0][1] - half_rf_size_w) + " " + Integer.toString(inputSize[0][0] - half_rf_size_h));
            for (int width = half_rf_size_w; width < inputSize[0][1] - half_rf_size_w; width += rfStride[1]) {
                for (int height = half_rf_size_h; height < inputSize[0][0] - half_rf_size_h; height += rfStride[0]) {
                    where_count++;
                }
            }


            rfIdLoc = new int[where_count][2];
            int id = 0;
            for (int width = half_rf_size_w; width < inputSize[0][1] - half_rf_size_w; width += rfStride[1]) {
                for (int height = half_rf_size_h; height < inputSize[0][0] - half_rf_size_h; height += rfStride[0]) {
                    rfIdLoc[id][0] = height;
                    rfIdLoc[id][1] = width;
                    id++;
                }
            }
        }
        else{
            rfIdLoc = new int[1][2];
            rfIdLoc[0][0] = 0;
            rfIdLoc[0][1] = 0;

        }
        return rfIdLoc;
    }

    //added by jacob for passing to activity
    public float[] getHiddenResponse (int layer) {

        return network.getHiddenResponse(layer);
    }

    //added by jacob for passing to activity
    public float getFiringNeuronAge (int layer, int neuron){

        return network.getFiringNeuronAge(layer, neuron);
    }

    //added by jacob for passing to activity
    public int getFiringNeuronType (int layer, int neuron){
        return network.getFiringNeuronType(layer, neuron);
    }

    //added by jacob for passing to activity
    public float[] getTopDownWeights (int layer, int neuron){

        return network.getTopDownWeights(layer, neuron);
    }

    //added by jacob for passing to activity
    public float[] getLateralWeights (int layer, int neuron){

        return network.getLateralWeights(layer, neuron);
    }

    //added by jacob for passing to activity
    public float[] getBottomUpWeights (int layer, int neuron){

        float[] weights = network.getBottomUpWeights(layer, neuron);
        return weights;
    }

    //added by jacob for initialization neuron types
    public int[] getInitializeNeuronInfo (){

        return network.getInitializeNeuronInfo();
    }

    //added by jacob for winner flags
    public boolean[] getWinnerFlags(int layer) {

        return network.getWinnerFlags(layer);
    }

    /**
     * Sets the response type for the hidden layer.
     * Triangle: Neuron responses set based on rank in top-k competition.
     * Goodness of Match: Neuron responses set based on relative value of pre-response.
     * @param responseType The chosen hidden layer response type.
     */
    public void setHiddenResponseType(RESPONSE_CALCULATION_TYPE responseType) {
        hiddenResponseType = responseType;
    }

    /**
     * Sets the response type for the motor layer.
     * Triangle: Neuron responses set based on rank in top-k competition.
     * Goodness of Match: Neuron responses set based on relative value of pre-response
     * @param responseType The chosen motor layer response type.
     */
    public void setMotorResponseType(RESPONSE_CALCULATION_TYPE[] responseType) {
        motorResponseType = responseType;
    }

//    // TODO: Document what concept is used for.
//    /**
//     * Sets the concept for the DN.
//     * @param concept The concept to be set.
//     */
//    public void setConcept(boolean concept) {
//        if(network != null)
//            network.setConcept(concept);
//    }
//
//    // TODO: Document what this does.
//    /**
//     * Sets learning.
//     * @param learning Learning.
//     * @param index Index.
//     */
//    public void setLearning(boolean learning, int index) {
//        if(network != null)
//            network.setLearning(learning, index);
//    }
//
//    // TODO: Document what this does.
//    /**
//     *
//     */
//    public void finish()
//    {
//        if(network != null)
//            network.finish();
//    }

    public void reset(){
        firstRun = true;
        initialized = false;
    }

    public ArrayList<byte[][]> getMotorWeights()
    {
        ArrayList<float[][]> weights = network.getMotorWeights();

        for(int motor = 0; motor < weights.size(); motor++)
        {
            for(int value = 0; value < weights.get(motor).length; value++)
            {
                weights.get(motor)[value] = normalize(weights.get(motor)[value]);
            }
        }

        ArrayList<byte[][]> weightsToReturn = new ArrayList<>();
        for(float[][] motor : weights)
        {
            byte[][] byteWeights = new byte[motor.length][motor[0].length];
            for(int i = 0; i < motor.length; i++)
            {
                for(int j = 0; j < motor[0].length; j++)
                {
                    byteWeights[i][j] = (byte)(255*motor[i][j]);
                }
            }
            weightsToReturn.add(byteWeights);
        }

        return weightsToReturn;
    }

    private float[] normalize(float[] input)
    {
        float[] result = new float[input.length];
        float min = input[0];
        float max = input[0];

        for (int i = 0; i < input.length; i++)
        {
//            input[i] = Math.abs(input[i]);
            if (input[i] < min)
            {
                min = input[i];
            }
            else if (input[i] > max)
            {
                max = input[i];
            }
        }

        if ((max - min) != 0) {
            for (int i = 0; i < input.length; i++) {
                result[i] = (input[i] - min) / (max - min);
            }
        }
        return result;
    }

    public float[][] getSensorGrid(){return network.getSensorGrid();}

    public int getNumberRfs(){
        return network.getNumberRfs();
    }

    public int getNumberNeurons(){
        return numNeuronsPassed;
    }
    
    public int[][] getRfSizes(){
        return rfSizes;
    }
    public int[][] getRfStrides(){
        return rfStrides;
    }
    public int[][] getInputSize(){
        return inputSize;
    }

}

package dn2;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import static dn2.HiddenLayer.MACHINE_FLOAT_ZERO;
//import MazeInterface.Commons;
//import MazeInterface.Commons.env;
//import MazeInterface.DNCaller;

/**
 * <h1>DN2</h1>
 *
 * <p>
 * This class implements the DN2 Neural Network Architecture
 * It takes user Sensor and Motor inputs, update the network's neuron weights,
 * and returns computed actions to the user.
 *
 * Every area of X,Y, and Z only update once.
 * Always from left column to right column as shown below.
 *
 *
 * Z  -	Z'
 *	 \ /
 *	 / \
 * Y  -	Y'
 * 	   /
 * 	 /
 * X 	X'
 * </p>
 *
 */
public class DN2 implements Serializable{

	// private -> Other classes outside the DN1 cannot see or change this variable unless DN1 allows it through methods calls
	// static -> Every DN1 instance created will have the same value
	// final -> Once initialized,it cannot be changed

	/**
	 * Class serial ID.
	 *
	 * Identifies the array of bytes of the object sent across network sockets or text files.
	 */
	private static final long serialVersionUID = 1;

	/*
	 * Sensor Variables
	 */

	/** Number of sensor (X) areas used by the network. */
	protected int numSensor;

	/** Array of SensorLayer instances used by the network. */
	protected SensorLayer[] sensor;

	// Pain and sweet sensors
	private SensorLayer sweetSensor;
	private SensorLayer painSensor;

	/*
	 * Motor Variables
	 */

	/** Number of motor (Z) areas used by the network. */
	protected int numMotor;

	/** Array of MotorLayer instances used by the network. */
	protected MotorLayer[] motor;

	/** Weight normalization frequency for Z neuron's bottom-up weights. */
	private int ZbottomupFrequency;

	/*
	 * Hidden Variables - Y area in DN
	 */

	/** TODO: COMPLETE THE COMMENT */
	private int numPrimaryHidden;


	/** TODO: COMPLETE THE COMMENT */
	protected PrimaryHiddenLayer[] prihidden;

	/** Number of hidden (Y) areas used by the network. */
	private int numHidden;

	/** Array of HiddenLayer instances used by the network. */
	protected HiddenLayer[] hidden;

	/** Number of neurons available for each HidenLayer instance. */
	private int[] numHiddenNeurons;

	/** Number of k winners for each HiddenLayer instance. */
	private int[] mHiddentopk;

	/** TODO: COMPLETE THE COMMENT */
	protected int[] lateralZone;

	/** TODO: COMPLETE THE COMMENT */
	private float lateralPercent;

	public boolean performanceDN;

	/** Write network's outputs into files */
	public FileWriter fw;

	// Nam added
	public void setSynapticMaintenanceAge(int age){
		for (int i=0;i<numHidden;i++){
			for(int j=0;j<hidden[i].hiddenNeurons.length;j++){
				hidden[i].hiddenNeurons[j].setSynapticMaintenanceAge(age);
			}
		}

	}
	/** Task indicator type variable */
	/*
	 * Group:
	 * MAZE:
	 * SPEECH:
	 */
	public enum MODE {GROUP, MAZE, Speech, NONE}

	/** Task indicator value */
	private MODE mode;

	public ArrayList<Long> bottomUpResponseTimes = new ArrayList<>();
	public ArrayList<Long> topDownResponseTimes = new ArrayList<>();
	public ArrayList<Long> hebbianLearnTime = new ArrayList<>();

	float[] maxVolumes;

	static boolean DEBUG = false;

	float[][] sensorGrid;

	private int[][] rfSizes; //receptive field size //default to full sensor
	private int[][] rfStrides; //receptive field stride //default to (1,1)

	public int numColumns;
	public int numRows;
	public int numBoxes;

	float[][] allHiddenInput;
	int[] allHiddenSize;

	//construct DN

	/**
	 * This constructor method initializes the network
	 * with all the required parameters
	 *
	 * @param numInput Total number of sensors in the network.
	 * @param inputSize Array of n-dimensions for each sensor area.
	 * @param numMotor Total number of motors in the network.
	 * @param motorSize Array of m-dimensions for each motor area.
	 * @param topKMotor Array of topK value for each motor area.
	 * @param numHidden Total number of hidden areas in the network.
	 * @param rfSizes The sizes of the receptive fields.
	 * @param rfStrides The number of shifts the receptive fields will do.
	 * @param rf_id_loc TODO: Ask Xiang
	 * @param numHiddenNeurons Number of hidden neurons per Hidden area.
	 * @param prescreenPercent TODO: Ask Xiang
	 * @param typeNum Starting number of hidden neurons for each DN2 type.
	 * @param growthrate Array of growth rate for each type, depending on available neurons.
	 * @param meanvalue TODO: Ask Xiang
	 * @param lateralpercent TODO: Ask Xiang
	 * @param dynamicInhibition Flag to determine whether dynamic Inhibition is used.
	 * @param lateral_zone TODO: Ask Xiang
	 * @param lateral_length TODO: Ask Xiang
	 * @param network_mode Indicates whether the network runs the Group task (Toy Data) or Maze task.
	 */
	public DN2(int numInput, int[][] inputSize, int numMotor, int[][] motorSize, int[] topKMotor,
			   int numHidden, int[][] rfSizes, int[][] rfStrides, int[][] rf_id_loc, int[] numHiddenNeurons,
			   float prescreenPercent, int[] typeNum, float[][] growthrate, float[][] meanvalue,
			   float lateralpercent, boolean dynamicInhibition, int[] lateral_zone, int lateral_length, MODE network_mode,
			   float[][] neuronGrowth, DNHandler.RESPONSE_CALCULATION_TYPE[] motorResponseType,
			   DNHandler.RESPONSE_CALCULATION_TYPE hiddenResponseType){

		// Initialize the layers
		this.mode = network_mode;

		// Initialize the sensor layers
		this.numSensor = numInput;
		sensor = new SensorLayer[numSensor]; // 2 for serotonin and dopamine buttons

		for (int i = 0; i < numSensor; i++) {
			sensor[i] = new SensorLayer(inputSize[i][0], inputSize[i][1]);
		}

		// TODO: Add these to the array above
		/*sweetSensor = new SensorLayer(1, 1);
		sensor[sensor.length-2] = sweetSensor;
		painSensor = new SensorLayer(1, 1);
		sensor[sensor.length-1] = painSensor;*/

		//pass receptive fields to local variables
		this.rfSizes = rfSizes;
		this.rfStrides = rfStrides;

		//compute numColumns, numRows, numBoxes
		numBoxes = 0;
		numRows = 0;
		numColumns = 0;

		int[] index = {0,0}; //x,y

		for (int i = 0; i < numSensor; i++){
			int width = rfSizes[i][1];
			int height = rfSizes[i][0];
			int stridex = rfStrides[i][1];
			int stridey = rfStrides[i][0];

			while (index[1] + stridey <= inputSize[i][0]){
				while (index[0] + stridex <= inputSize[i][1]){
					index[0] += stridex;
					numColumns++;
				}
				index[1] += stridey;
				numRows++;
			}
			index = new int[]{0,0};
			numBoxes = numBoxes + numRows * numColumns;
			numRows = 0;
			numColumns = 0;
		}

		int boxNumber = 0;

		index = new int[]{0,0}; //x,y

		int[] sensorSizeSeperated = new int[numBoxes];;

		for (int i = 0; i < numSensor; i++){
			int width = rfSizes[i][1];
			int height = rfSizes[i][0];
			int stridex = rfStrides[i][1];
			int stridey = rfStrides[i][0];

			int iter = 0;

			while (index[1] + stridey <= inputSize[i][0]){
				while (index[0] + stridex <= inputSize[i][1]){
					sensorSizeSeperated[boxNumber] = height*width;
					boxNumber++;
					index[0] += stridex;
				}
				index[1] += stridey;
				index[0] = 0;
			}

			index = new int[]{0,0};
		}

		//set z neuron's bottom-up weight normalization frequency
		// calculate total sizes of the sensor
		int totalSensor = totalSize(inputSize);
		// calculate total sizes of the motor
		int totalMotor = totalSize(motorSize);
		//get the number of y neurons
		this.numHiddenNeurons = numHiddenNeurons;

		this.numMotor = numMotor;

		// Initialize the hidden layers
		//get the top-k's k value
		this.numPrimaryHidden = numPrimaryHidden;
		//get the number of y layers
		this.numHidden = numHidden;
		//initialize the hidden-layer array
		hidden = new HiddenLayer[numHidden];
		//initialize each component of hidden-layer array
		this.lateralPercent = lateralpercent;

		// Initialize the Primary hidden layer
		int totalPriHidden = 0;
		if(numPrimaryHidden != 0) {
			prihidden = new PrimaryHiddenLayer[numPrimaryHidden];
			for (int i = 0; i < numPrimaryHidden; i++) {
				//prihidden[i] = new PrimaryHiddenLayer(4, topKHidden[i], totalSensor, totalMotor, inputSize, 4, meanvalue);
				totalPriHidden = totalSize(prihidden);
			}
		}

		// Initialize the hidden layer
		for (int i = 0; i < numHidden; i++) {
			int numPriLateral = 0;
			for(int j = 0; j < numPrimaryHidden; j++) {
				numPriLateral += prihidden[j].numNeurons*prihidden[j].mDepth;
			}
			hidden[i] = new HiddenLayer(numHiddenNeurons[i], totalSensor, sensorSizeSeperated, totalMotor,
					rfSizes, rfStrides, rf_id_loc, inputSize, prescreenPercent,
					typeNum, growthrate, meanvalue, dynamicInhibition, neuronGrowth, hiddenResponseType, numBoxes, numMotor);
		}

		// The total number of Hidden Neurons
		int totalHidden = totalSize(hidden);

		// Total number of Y neurons.
		int totalLateral = totalPriHidden+totalHidden;

		// Initialize the motor layers
		motor = new MotorLayer[numMotor];

		lateralZone = lateral_zone;

		int lateralSize = 0; //added so that lateral motor connections see all neurons initialize
		for (int i = 0; i < numMotor; i++) {
			lateralSize += motorSize[i][0]*motorSize[i][1];
		}

		// Create a motor layer for each motor
		for (int i = 0; i < numMotor; i++) {
			// Determine if the current motor is within the Lateral Zone
			boolean contains = arrayContains(lateral_zone, i);
			if (!contains){
//			    motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, motorSize[i][0]*motorSize[i][1], motorResponseType[i]); // Was lateral size 0
				motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, lateralSize, motorResponseType[i]); // Was lateral size 0

			} else {
//				motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, motorSize[i][0]*motorSize[i][1], motorResponseType[i]); // Was lateral size lateral_length
				motor[i] = new MotorLayer(motorSize[i][0], motorSize[i][1], topKMotor[i], totalLateral, lateralSize, motorResponseType[i]); // Was lateral size lateral_length
			}
		}


		// TODO: This should go in initialize rf
		//3x3 mask
		/*for(int maskIdx = 0;maskIdx<9;maskIdx++) {
			int row = maskIdx/3;
			int col = maskIdx%3;

			float mask[] = new float[inputSize[0][1]];
			for(int i=0;i<inputSize[0][1];i++) {
				mask[i] = 0;
			}

			for(int im_idx = 0;im_idx<2;im_idx+=1) {
				for(int r=0;r<cellSize;r++) {
					for(int c=0;c<cellSize;c++) {
						for(int color = 0;color<3;color++) {
							int inputIdx = im_idx*(3*3*cellSize*cellSize*3)+(row*cellSize+r)*(cellSize*3*3)+(col*cellSize+c)*3+color;
							// For with alpha value (change color to color<4)
							//int inputIdx = im_idx*(3*3*cellSize*cellSize*4)+(row*cellSize+r)*(cellSize*4*3)+(col*cellSize+c)*4+color;
							mask[inputIdx] = 1.f;
						}
					}
				}
			}

			for(int i = 0;i <numHiddenNeurons[0];i++) {
				int neuronIdx = maskIdx*numHiddenNeurons[0]+i;
				setHiddenBottomupMask(0,neuronIdx,mask);
			}
		}*/

		setMotorisTopk(true);

		maxVolumes = new float[9];

	}


	public void setTopDownWeights(float topDown){
		for (int i=0;i<numHidden;i++){
			for(int j=0;j<hidden[i].hiddenNeurons.length;j++){
				hidden[i].hiddenNeurons[j].setTopDownWeights(topDown);
			}
		}
	}


	//Created by Arden 5/15/2019 to show DN properties when DN is loaded or created
	public boolean showDN2() {
		System.out.println(this.numHidden);
		System.out.println(this.lateralPercent);
		for(int i=0; i<lateralZone.length; i++) {
			System.out.print(lateralZone[i]);
		}
		System.out.println();
		System.out.println(ZbottomupFrequency);
		return true;
	}

	/**
	 * This method sets the growth rate of each Hidden neurons types.
	 * @param growth_rate Array of growth rate for each type, depending on available neurons.
	 */
	public void setGrowthRate(float[][] growth_rate){
		for (int i = 0; i < numHidden; i++){
			hidden[i].setGrowthRate(growth_rate);
		}
	}

	/**
	 * This method sets the growth rate table for the DN and tells when to initialize new neurons.
	 * @param neuron_growth_rate Array of growth rate for each type, depending on available neurons.
	 */
	public void setNeuronGrowthRate(float[][] neuron_growth_rate){
		for (int i = 0; i < numHidden; i++){
			hidden[i].setNeuronGrowthRate(neuron_growth_rate);
		}
	}

	/**
	 * This method checks whether a target integer is within an array of integers.
	 *
	 * @param array Array of integer values.
	 * @param target Integer value to be searched.
	 * @return result True if the target is within the array.
	 */
	public boolean arrayContains(int[] array, int target){
		boolean result = false;
		for (int i = 0; i < array.length; i++) {
			if (array[i] == target) {
				result = true;
				break;
			}
		}
		return result;
	}

	/**
	 * This method connect the user sensor input to the sensor on the specified index.
	 *
	 * @param index SensorLayer that receives the input.
	 * @param input The user sensor input.
	 */
	public void setSensorInput(int index, float[][] input){
		sensor[index].setInput(input);
	}

	/**
	 * This method connect the user motor input to the motor on the specified index.
	 *
	 * @param index MotorLayer that receives the input.
	 * @param input The user motor input.
	 */
	public void setMotorInput(int index, float[][] input){
		motor[index].setInput(input);
	}

	/**
	 * This method returns the HiddenLayer instance at the specified index i.
	 *
	 * @param i HiddenLayer index
	 * @return HiddenLayer Instance at index i
	 */
	public HiddenLayer getHiddenLayer(int i){
		assert(i < hidden.length); // validates the index is within range.
		return hidden[i];
	}

//	/**
//	 *
//	 * @param i
//	 * @param j
//	 * @param mask
//	 */
//    public void setHiddenBottomupMask(int i, int j, float[] mask){
//    	hidden[i].setBottomupMask(mask, j);
//    }

//    /**
//     *
//     * @param i
//     * @param j
//     * @param mask
//     */
//    public void setHiddenTopdownMask(int i, int j, float[] mask){
//    	hidden[i].setTopdownMask(mask, j);
//    }

	/**
	 *
	 * @param sensorInput
	 * @param learn_flag
	 */
	public void computePrimaryHiddenResponse(float[][][] sensorInput, boolean learn_flag){

		// set the input and motors
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput[i]);

			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}

		// computes the new response for Y
		for (int i = 0; i < numPrimaryHidden; i++) {
			prihidden[i].computeBottomUpResponse(allSensorInput, allSensorSize);

			// local input
			prihidden[i].computeResponse(learn_flag, mode);

/*			System.out.println("HiddenResponse");
			displayResponse(hidden[i].getResponse1D());
*/

			if (learn_flag)
				prihidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize));
		}

	}

	/**
	 *
	 */
	public void savePriBottomWeights(){
		for (int i = 0; i < numPrimaryHidden; i++) {
			prihidden[i].saveWeightToFile("PriHidden"+Integer.toString(i));
		}
	}

	/**
	 *
	 * @param i
	 * @param j
	 * @return
	 */
	public boolean getPrilearningflage(int i, int j) {
		return prihidden[0].hiddenNeurons[i][j].getState();
	}

	/**
	 *
	 * @return
	 */
	public float sumPrinewresponses() {
		float[][] allHiddenInput = new float[numPrimaryHidden][];
		float summ = 0;
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getNewResponse1D();
			for(int j = 0; j < allHiddenInput[i].length; j++) {
				if(allHiddenInput[i][j] != 0) {
					summ += allHiddenInput[i][j];
				}
			}
		}
		return summ;
	}

	/**
	 *
	 * @return
	 */
	public float sumPrioldresponses() {
		float[][] allHiddenInput = new float[numPrimaryHidden][];
		float summ = 0;
		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(1.0f);
			for(int j = 0; j < allHiddenInput[i].length; j++) {
				summ += allHiddenInput[i][j];
			}
		}
		return summ;
	}

	/**
	 * This method initializes some parameters to the DNCaller class,
	 * calls an overloaded version of computeHiddenResponse.
	 *
	 * @param sensorInput Array of 2D inputs for each sensor
	 * @param motorInput Array of 2D inputs for each motor
	 * @param learn_flag Indicate whether this method was called during the learning stage
	 * @param current_type TODO: Ask Xiang
	 * @param current_loc TODO: Ask Xiang
	 * @param current_scale TODO: Ask Xiang
	 */
	/*public void computeHiddenResponse(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag,
			env current_type, int current_loc, int current_scale) {
		DNCaller.curr_loc = current_loc;
		DNCaller.curr_scale = current_scale;
		DNCaller.curr_type = current_type.ordinal();
		computeHiddenResponse(sensorInput, motorInput, learn_flag, current_loc, current_scale);
	}	*/

	/**
	 * This method initializes some parameters to the DNCaller class,
	 * calls an overloaded version of computeHiddenResponse.
	 *
	 * @param sensorInput Array of 2D inputs for each sensor
	 * @param motorInput Array of 2D inputs for each motor
	 * @param learn_flag Indicate whether this method was called during the learning stage
	 */
	public void computeHiddenResponse(ArrayList<float[][]> sensorInput, ArrayList<float[][]> motorInput, boolean learn_flag){
		/*if (mode == DN2.MODE.MAZE){
			DNCaller.curr_loc = -1;
			DNCaller.curr_scale = -1;
			DNCaller.curr_scale = -1;
		}*/
		computeHiddenResponse(sensorInput, motorInput, learn_flag, -1, -1);
	}

//	/**
//	 *
//	 * @param sensorInput
//	 * @param motorInput
//	 * @param learn_flag
//	 */
//	public void computeHiddenResponseParallel(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag){
//
//		computeHiddenResponseParallel(sensorInput, motorInput, learn_flag, -1, -1);
//	}

	/**
	 * This method computes the bottom-up, top-down, and lateral weights of the Y areas.
	 *
	 * @param sensorInput Array of 2D inputs for each sensor
	 * @param motorInput Array of 2D inputs for each motor
	 * @param learn_flag Indicate whether this method was called during the learning stage
	 * @param rf_loc TODO: Ask Xiang
	 * @param rf_size TODO: Ask Xiang
	 */
	public void computeHiddenResponse(ArrayList<float[][]> sensorInput, ArrayList<float[][]> motorInput, boolean learn_flag, int rf_loc, int rf_size){

		// Initialize the master sensorInput and sensorSize arrays
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];

		/*
		 * 1) Set the user input to each sensor.
		 * 2) Append the sensor input (converted to 1D array) into a 1D master array.
		 * 3) Append the size of sensor into a master sensorSize array.
		 */
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput.get(i));
			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}

		// Initialize the master motorInput and motorSize arrays
		float[][] allMotorInput = new float[numMotor][];
		int[] allMotorSize = new int[numMotor];

		/*
		 * 1) Set the user input to each motor.
		 * 2) Append the motor input (converted to 1D array) into a 1D master array.
		 * 3) Append the size of motor into a master motorSize array.
		 */
		for (int i = 0; i < numMotor; i++) {
			motor[i].setInput(motorInput.get(i));
			allMotorInput[i] = motor[i].getInput1D();
			allMotorSize[i] =  allMotorInput[i].length;
		}

		/*
		 * 1) Compute the BottomUp response (connection with X).
		 * 2) Compute the TopDown response (connection with Z).
		 * 3) Compute the Lateral response (connection with Y).
		 */

		sensorGrid = new float[numBoxes][]; // +8 for serotonin and dopamine
		int boxNumber = 0;

		int[] index = {0,0}; //x,y

		for (int i = 0; i < numSensor; i++){
			int width = rfSizes[i][1];
			int height = rfSizes[i][0];
			int stridex = rfStrides[i][1];
			int stridey = rfStrides[i][0];

			while (index[1] + stridey <= sensorInput.get(i).length){
				while (index[0] + stridex <= sensorInput.get(i)[0].length){
					float[] receptiveField = new float[height*width + 1 + numMotor * 2]; // numMotor * 2 reinforcement

					int xind = index[0];
					int yind = index[1];

					int it = 0;

					for (int b = 0; b < stridey; b++){
						for (int a = 0; a < stridex; a++){
							receptiveField[it] = sensorInput.get(i)[yind][xind];
							it++;
							xind++;
						}
						yind++;
						xind = index[0];
					}

					receptiveField = epsilon_normalize(receptiveField, receptiveField.length, i); //epsilon normalize rf
//					System.out.println(Arrays.toString(receptiveField));
					
					for (int w = 1; w < numMotor + 1; w++){
						receptiveField[receptiveField.length - w] = sensorInput.get(numSensor)[0][numMotor - w]; //pain
						receptiveField[receptiveField.length - w - 1] = sensorInput.get(numSensor+1)[0][numMotor - w]; //sweet
					}

					sensorGrid[boxNumber] = receptiveField;
					boxNumber++;
					index[0] += stridex;
				}
				index[1] += stridey;
				index[0] = 0;
			}

			index = new int[]{0,0};
		}

		float[] motor = inputToWeights(allMotorInput, allMotorSize);

		epsilon_normalize(motor, motor.length);

		float[][] responses = new float[hidden[0].numBoxes][hidden[0].numNeuronPerColumn];
		for(int cl = 0; cl < hidden[0].numBoxes; cl++) {
			for(int j = 0; j < hidden[0].numNeuronPerColumn; j++) {
				responses[cl][j] = hidden[0].hiddenNeurons[hidden[0].numNeuronPerColumn*cl + j].getoldresponse();
			}
		}


		for(int cl = 0; cl < hidden[0].numBoxes; cl++)
		{
			float norm = 0.0f;
			for(int i = 0; i < responses[cl].length; i++)
			{
				norm += (responses[cl][i] * responses[cl][i]);
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;

			for(int i = 0; i < responses[cl].length; i++)
			{
				responses[cl][i] /= norm;
			}
		}


		// computes the new response for Y
		for (int i = 0; i < numHidden; i++) {
			// Inner product between Y area bottom-up weights and Sensor inputs
			hidden[i].computeBottomUpResponse(sensorGrid, allSensorSize);

			// Inner product between Y area top-down weights and Motor inputs
			hidden[i].computeTopDownResponse(motor, allMotorSize);


			// Determines if the current neuron is Primary Neuron
			if(hidden[i].isPriLateral) {

				// Initialize the HiddenInput of the primary neurons
				float[][] allHiddenInput = new float[numPrimaryHidden][];
				int[] allHiddenSize = new int[numPrimaryHidden];

				// Compute the Primary neurons' response
				for (int j = 0; j < numPrimaryHidden; j++) {
					allHiddenInput[j] = prihidden[i].getResponse1D(1.0f);
					allHiddenSize[j] =  allHiddenInput[i].length;
				}

				// Update the Lateral vector of the hidden layer
				hidden[i].setPriLateralvector(inputToWeights(allHiddenInput, allHiddenSize));
			}

			// Inner product between Y area lateral weights and Hidden inputs
			hidden[i].computeLateralResponse(responses);

			/*
			 * Location Motor Attention Component for DN2.
			 *
			 * Each problem has a different attention schema.
			 *
			 * A) DN Maze -> Information is already defined when this task calls
			 * 				 the DN2 to compute Hidden responses.
			 *
			 * B) Group -> Following the Toy problem, our focus of attention is the
			 * 			   the first location motor neuron that we detect to be firing.
			 *
			 * C) Default -> -1 means that the DN2 does not have attention to location motors.
			 */
			int where_id = -1;
			if (mode == MODE.MAZE){
				where_id = 0; // information already stored in DNCaller.
			}
			if(mode == MODE.GROUP){
				for (int j = 0; j < allMotorInput[1].length-1; j++){
					if (allMotorInput[1][j] == 1){
						where_id = j;
						break;
					}
				}
			}

			/// local input
			// The where_id determines the spatial attention TODO: Ask Xiang
			hidden[i].computeResponse(where_id, learn_flag, mode);
			
			// Outputs the hidden responses to the console.
			if(DEBUG) {
				System.out.println("HiddenResponse");
				displayResponse(hidden[i].getResponse1D());
			}

			if (learn_flag) {
				hidden[i].hebbianLearnHidden(sensorGrid,motor,responses);
			}
			else { // Otherwise, increase their firing ages.
				hidden[i].addFiringAges();
			}
		}

	}



//	/**
//	 *
//	 * @param sensorInput Array of 2D inputs for each sensor
//	 * @param motorInput Array of 2D inputs for each motor
//	 * @param learn_flag Indicate whether this method was called during the learning stage
//	 * @param rf_loc
//	 * @param rf_size
//	 */
//	public void computeHiddenResponseParallel(float[][][] sensorInput, float[][][] motorInput, boolean learn_flag, int rf_loc, int rf_size){
//
//		// Initialize the master sensorInput and sensorSize arrays
//		float[][] allSensorInput = new float[numSensor][];
//		int[] allSensorSize = new int[numSensor];
//
//		/*
//		 * 1) Set the user input to each sensor.
//		 * 2) Append the sensor input (converted to 1D array) into a 1D master array.
//		 * 3) Append the size of sensor into a master sensorSize array.
//		 */
//		for (int i = 0; i < numSensor; i++) {
//			sensor[i].setInput(sensorInput[i]);
//			allSensorInput[i] = sensor[i].getInput1D();
//			allSensorSize[i] =  allSensorInput[i].length;
//		}
//
//		// Initialize the master motorInput and motorSize arrays
//		float[][] allMotorInput = new float[numMotor][];
//		int[] allMotorSize = new int[numMotor];
//
//		/*
//		 * 1) Set the user input to each motor.
//		 * 2) Append the motor input (converted to 1D array) into a 1D master array.
//		 * 3) Append the size of motor into a master motorSize array.
//		 */
//		for (int i = 0; i < numMotor; i++) {
//			motor[i].setInput(motorInput[i]);
//			allMotorInput[i] = motor[i].getInput1D();
//			allMotorSize[i] =  allMotorInput[i].length;
//		}
//
//		/*
//		 * 1) Compute the BottomUp response (connection with X).
//		 * 2) Compute the TopDown response (connection with Z).
//		 * 3) Compute the Lateral response (connection with Y).
//		 */
//
//		// computes the new response for Y
//		for (int i = 0; i < numHidden; i++) {
//			// Inner product between Y area bottom-up weights and Sensor inputs
//			hidden[i].computeBottomUpResponseInParallel(allSensorInput, allSensorSize);
//
//			// Inner product between Y area top-down weights and Motor inputs
//			hidden[i].computeTopDownResponse(allMotorInput, allMotorSize);
//
//			// Inner product between Y area lateral weights and Hidden inputs
//			hidden[i].computeLateralResponse();
//
//			int where_id = -1;
//
//			// local input
//			// There is no attention to the parallel response.
//			hidden[i].computeResponse(where_id, learn_flag, mode);
//
//			// Outputs the hidden responses to the console.
//			System.out.println("HiddenResponse");
//			displayResponse(hidden[i].getResponse1D());
//
//			// Update Y area weights if we observe new inputs.
//			if (learn_flag) {
//				hidden[i].hebbianLearnHiddenParallel(allSensorInput, inputToWeights(allSensorInput, allSensorSize),inputToWeights(allMotorInput, allMotorSize));
//			}
//			else { // Otherwise, increase their firing ages.
//				hidden[i].addFiringAges();
//			}
//		}
//
//	}

	/**
	 * Computes the hiddenLayer response with just the updated bottomup input response.
	 * The motor input is just supervised, not used for hidden response computation.
	 * The Y neurons are updated, but the firing ages are not increased.
	 * TODO: Ask Xiang
	 *
	 * @param sensorInput Array of 2D inputs for each sensor
	 * @param motorInput Array of 2D inputs for each motor
	 * @param learn_flag Indicate whether this method was called during the learning stage
	 * @param type The hidden neuron type?
	 */
	public void computeHiddenResponse(float[][][] sensorInput,  float[][][] motorInput, boolean learn_flag, int type){

		// Initialize the master sensorInput and sensorSize arrays
		float[][] allSensorInput = new float[numSensor][];
		int[] allSensorSize = new int[numSensor];

		/*
		 * 1) Set the user input to each sensor.
		 * 2) Append the sensor input (converted to 1D array) into a 1D master array.
		 * 3) Append the size of sensor into a master sensorSize array.
		 */
		for (int i = 0; i < numSensor; i++) {
			sensor[i].setInput(sensorInput[i]);
			allSensorInput[i] = sensor[i].getInput1D();
			allSensorSize[i] =  allSensorInput[i].length;
		}

		// Initialize the master motorInput and motorSize arrays
		float[][] allMotorInput = new float[numMotor][];
		int[] allMotorSize = new int[numMotor];

		/*
		 * 1) Set the user input to each motor.
		 * 2) Append the motor input (converted to 1D array) into a 1D master array.
		 * 3) Append the size of motor into a master motorSize array.
		 */
		for (int i = 0; i < numMotor; i++) {
			motor[i].setInput(motorInput[i]);
			allMotorInput[i] = motor[i].getInput1D();
			allMotorSize[i] =  allMotorInput[i].length;
		}

		/*
		 * 1) Compute the BottomUp response (connection with X).
		 */

		// computes the new response for Y
		for (int i = 0; i < numHidden; i++) {

			hidden[i].computeBottomUpResponse(allSensorInput, allSensorSize);
			int where_id = -1;

			// local input
			hidden[i].computeResponse(where_id, learn_flag, mode, type);

			// Outputs the hidden responses to the console.
//			System.out.println("HiddenResponse");
//			displayResponse(hidden[i].getResponse1D());

			// Update Y area weights if we observe new inputs.
			// This one does not increase the firing ages
			//if (learn_flag)
			//hidden[i].hebbianLearnHidden(inputToWeights(allSensorInput, allSensorSize),
			//		                     inputToWeights(allMotorInput, allMotorSize));
		}

	}

	/**
	 * Displays an array of float values to console.
	 *
	 * @param r Hidden response array
	 */
	private void displayResponse(float[] r) {
		if(DEBUG) {
			System.out.print(r[0]);
			for (int j = 1; j < r.length; j++) {
				System.out.print("," + r[j]);
			}
			System.out.println();
		}
	}

	/**
	 * Compute motor lateral response only, for planning in DN2.
	 * Z to Z connection response.
	 *
	 * @param lateral_input TODO: Ask Xiang.
	 * @param motorIndex Motor to compute lateral response.
	 * @return response Motor lateral response.
	 */
	public float[][] computeMotorLateralResponse (float[] lateral_input, int motorIndex){
		float[][] response;
		motor[motorIndex].computeLateralResponse(lateral_input);
		response = motor[motorIndex].getLateralResponse2D();
		return response;
	}

	/**
	 * Compute motor response only for a single motor, no update for the weights.
	 *
	 *
	 * @param motorIndex The motor to compute response.
	 * @return response Computed motor response.
	 */
	public float[][] computeMotorResponse(int motorIndex, float pain, float sweet){

		float[][] response;

		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];

		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.9f);
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			// Add serotonin and dopamine input
			allHiddenInput[i][allHiddenSize[i]-2] = pain;
			allHiddenInput[i][allHiddenSize[i]-1] = sweet;
		}

		// once the final responses are set, compute the motor responses
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		response = motor[motorIndex].getNewMotorResponse2D();

		return response;
	}

	/**
	 *
	 * @param motorIndex The motor to compute response.
	 * @param learning_flag Flag to determine if the motor updates weights with hebbian learning
	 * @return The computed response.
	 */
	public float[][] computeMotorResponse(int motorIndex, boolean learning_flag){

		float[][] response;

		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];

		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.9f);
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		// once the final responses are set, compute the motor responses
		motor[motorIndex].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
		response = motor[motorIndex].getNewMotorResponse2D();

		if(learning_flag){
			motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
		}

		return response;
	}

	/**
	 * Compute motor response for all motors only, no update for the weights.
	 *
	 * @return response Computed motor response.
	 */
	public ArrayList<float[][]> computeMotorResponse(float[][] pain, float[][] sweet){
		ArrayList<float[][]> response = new ArrayList<>();

		// get all hiddenLayer responses
		// update the hidden inputs
		allHiddenInput = new float[numPrimaryHidden+numHidden][];
		allHiddenSize = new int[numPrimaryHidden+numHidden];

		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.9f);
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			// TODO: Find a more efficient method to pass serotonin and dopamine to motor
			float [] temp = hidden[i-numPrimaryHidden].getResponse1D();
			float [] temp2 = new float[temp.length + 2]; // +2 for serotonin and dopamine
			for (int k =0; k < temp.length; k++) {
				temp2[k] = temp[k];
			}
			allHiddenInput[i] = temp2; //hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
			// TODO: Make sure serotonin and dopamine are making it to the motors

			/*for (int j = 0; j < pain[0].length; j++)
			{
				allHiddenInput[i][allHiddenSize[i]-(2*numMotors) + j] = pain[0][j];
				allHiddenInput[i][allHiddenSize[i]-(numMotors) + j] = sweet[0][j];
			}*/
			// Add serotonin and dopamine input
			//allHiddenInput[i][allHiddenSize[i]-2] = pain;
			//allHiddenInput[i][allHiddenSize[i]-1] = sweet;
		}

		float[] motorResponses;
		int motorResponsesSize = 0;
		for (int i = 0; i < numMotor; i++) { //get needed size for lateral
			motorResponsesSize += motor[i].getWidth();
		}
		motorResponses = new float[motorResponsesSize];
		int it = 0;
		for (int i = 0; i < numMotor; i++) { //pass all motors for lateral
			for (int j = 0; j < motor[i].getMotorResponse().length; j++){
				motorResponses[it] = motor[i].getMotorResponse()[j];
				it++;
			}
		}

		// TODO: Make sure only disparity motors are being reinforced
		// once the final responses are set, compute the motor responses
		for (int i = 0; i < numMotor; i++) {
			//allHiddenInput[0][allHiddenSize[0]-2] = pain[0][2*i]; // [p0,s0,p1,s1,p2,s2,p3,s3]
			//allHiddenInput[0][allHiddenSize[0]-1] = sweet[0][2*i+1]; //TODO deleted by Jacob because it make no sense

			allHiddenInput[0][allHiddenSize[0]-2] = pain[0][i];
			allHiddenInput[0][allHiddenSize[0]-1] = sweet[0][i];

//			motor[i].computeLateralResponse(motor[i].getMotorResponse());
			motor[i].computeLateralResponse(motorResponses);

			motor[i].computeResponse(inputToWeights(allHiddenInput, allHiddenSize));
			response.add(i, motor[i].getNewMotorResponse2D());

//			allHiddenInput[0][allHiddenSize[0]-2] = 0; //setting to 0 makes no sense here.
//			allHiddenInput[0][allHiddenSize[0]-1] = 0;
			//allHiddenInput[0][allHiddenSize[0]-2] = 0;
			//allHiddenInput[0][allHiddenSize[0]-1] = 0;
		}

//		System.out.println(Arrays.deepToString("allHiddenInput: " + allHiddenInput));

		return response;


	}

	/**
	 * Set the inputs for all the motors. Update the motor weights of the DN.
	 *
	 * @param supervisedMotor 2D inputs for all motors
	 */
	public void updateSupervisedMotorWeights(ArrayList<float[][]> supervisedMotor){

		for (int i = 0; i < supervisedMotor.size(); i++) {
			motor[i].setSupervisedResponse(supervisedMotor.get(i));
		}

		float[] oldMotorResponses;
		int motorResponsesSize = 0;
		for (int i = 0; i < numMotor; i++) { //get needed size for lateral
			motorResponsesSize += motor[i].getWidth();
		}
		oldMotorResponses = new float[motorResponsesSize];
		int it = 0;
		for (int i = 0; i < numMotor; i++) { //pass all motors for lateral
			for (int j = 0; j < motor[i].getOldMotorResponse().length; j++){
				oldMotorResponses[it] = motor[i].getOldMotorResponse()[j];
				it++;
			}
		}

		updateMotorWeights();
	}

	/**
	 * Set the inputs for a single motor.
	 *
	 * @param motorIndex Index of the motor to be supervised.
	 * @param supervisedMotor 2D input for the motor.
	 */
	public void updateSupervisedMotorWeights(int motorIndex, float[][] supervisedMotor){

		motor[motorIndex].setSupervisedResponse(supervisedMotor);


		updateMotorWeights(motorIndex);
	}

	/**
	 * This method replaces the old primary hidden response values at time t-1
	 * with the computed primary hidden response values at time t.
	 */
	public void replacePriHiddenResponse(){
		for (int i = 0; i < prihidden.length; i++) {
			prihidden[i].replaceHiddenLayerResponse();
		}
	}

	/**
	 * This method replaces the old hidden response values at time t-1
	 * with the computed hidden response values at time t.
	 */
	public void replaceHiddenResponse(){
		for (int i = 0; i < hidden.length; i++) {
			hidden[i].replaceHiddenLayerResponse();
		}
	}

	/**
	 * This method replaces the old motor response values at time t-1
	 * with the computed motor response values at time t.
	 */
	public void replaceMotorResponse(){
		for (int i = 0; i < motor.length; i++) {
			motor[i].replaceMotorLayerResponse();
		}
	}

	/**
	 * Does the motor hebbian learning based on the computed response for all motors.
	 */
	public void updateMotorWeights(){

//		// get all hiddenLayer responses
//		// update the hidden inputs
//		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
//		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];
//
//		for (int i = 0; i < numPrimaryHidden; i++) {
//			allHiddenInput[i] = prihidden[i].getResponse1D(0.9f);
//			allHiddenSize[i] =  allHiddenInput[i].length;
//		}
//
//		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
//			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
//			allHiddenSize[i] =  allHiddenInput[i].length;
//		}
		for (int i = 0; i < numMotor; i++) {
			motor[i].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));

		}
	}

	/**
	 * Does the motor hebbian learning based on the computed response for a single motor.
	 *
	 * @param motorIndex Index of motor to be updated.
	 */
	public void updateMotorWeights(int motorIndex){
		// get all hiddenLayer responses
		// update the hidden inputs
		float[][] allHiddenInput = new float[numPrimaryHidden+numHidden][];
		int[] allHiddenSize = new int[numPrimaryHidden+numHidden];

		for (int i = 0; i < numPrimaryHidden; i++) {
			allHiddenInput[i] = prihidden[i].getResponse1D(0.9f);
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		for (int i = numPrimaryHidden; i < numPrimaryHidden+numHidden; i++) {
			allHiddenInput[i] = hidden[i-numPrimaryHidden].getResponse1D();
			allHiddenSize[i] =  allHiddenInput[i].length;
		}

		motor[motorIndex].hebbianLearnMotor(inputToWeights(allHiddenInput, allHiddenSize));
	}

	/**
	 * Update the lateral weights of a single motor.
	 *
	 * @param motorIndex Index of the motor to update
	 * @param response Lateral motor responses 1D array
	 * @param firing_neuron Index of the firing neuron
	 */
	public void updateLateralMotorWeights(int motorIndex, float[] response, int firing_neuron){
		/*if (arrayContains(lateralZone, motorIndex) && firing_neuron != Commons.NULLVALUE){
			motor[motorIndex].hebbianLearnLateral(response, firing_neuron);
		}*/
	}

	/* This function is never used. It should be removed
	private int maxIndex(float[] values){
		int index = (values[0] != 0.0f) ? 0:-1;

		float max = values[0];

		for (int i = 0; i < values.length; i++) {
			if(values[i] > max){
				max = values[i];
				index = i;
			}
		}

		return index;
	}
	*/

	/**
	 * This method converts a 2D input array into a 1D input weight array.
	 *
	 * @param input 2D input from the network.
	 * @param size Array of number of neurons per input (sensor/motor) area.
	 * @return weights Array of 1D weights.
	 */
	private float[] inputToWeights(float[][] input, int[] size){

		int total = 0;
		// compute the total size
		for (int i = 0; i < size.length; i++) {
			total += size[i];
		}

		float[] weights = new float[total];

		int beginIndex = 0;
		for (int j = 0; j < input.length; j++) {
			System.arraycopy(input[j], 0, weights, beginIndex, size[j]);
			beginIndex += size[j];
		}

		return weights;

	}

	/**
	 * This method returns the total number of primary neurons used by the Y areas.
	 *
	 * @param hidden2 Array of primary Y areas.
	 * @return total Number of primary Y neurons used across the entire network.
	 */
	protected int totalSize(PrimaryHiddenLayer[] hidden2){
		int total = 0;

		for (int i = 0; i < hidden2.length; i++) {
			total += (hidden2[i].getNumNeurons());
		}

		return total;
	}

	/**
	 * This method returns the total number of neurons used by the Y areas.
	 *
	 * @param hidden2 Array of Y areas.
	 * @return total Number of Y neurons used across the entire network.
	 */
	protected int totalSize(HiddenLayer[] hidden2){
		int total = 0;

		for (int i = 0; i < hidden2.length; i++) {
			total += (hidden2[i].getNumNeurons());
		}

		return total;
	}

	/**
	 * This method returns the total number of neurons used by the sensor/motor areas.
	 *
	 * @param size Array 2D input dimensions (height, width) for each motor.
	 * @return total Number of neurons used by the sensor/motor.
	 */
	protected int totalSize(int[][] size){
		int total = 0;

		for (int i = 0; i < size.length; i++) {
			total += (size[i][0] * size[i][1]);
		}

		return total;
	}

	/**
	 * Write the hidden and motor weights to text files. Helpful for visualization.
	 */
	// Probably make it to accept filenames instead of having fixed locations
	public void saveToText(){
		for (int i = 0; i < hidden.length; i++){
			hidden[i].saveWeightToFile("Audition_Data/hidden" + Integer.toString(i));
		}
		for (int i = 0; i < motor.length; i++){
			motor[i].saveWeightToFile("Audition_Data/motor" + Integer.toString(i));
		}
	}

	/**
	 * Write the motor weights to text files. Helpful for visualization.
	 */
	// Probably make it to accept filenames instead of having fixed locations
	public void saveMotorToText(){
		for (int i = 0; i < motor.length; i++){
			motor[i].saveWeightToFile("Audition_Data/motor" + Integer.toString(i));
		}
	}

	/**
	 * Writes the firing ages to text files. Helpful for visualization.
	 * @param ind
	 */
	public void saveAgeToText(int ind){
		for (int i = 0; i < hidden.length; i++){
			hidden[i].saveAgeToFile("Disparity_data/hidden" + Integer.toString(i), Integer.toString(ind));
		}
	}

	/**
	 * This function saves the DN2 object into a serial file.
	 *
	 * @param name Name for the serial file.
	 */
	public void serializeSave(String name) {
		try{
			FileOutputStream fileOut = new FileOutputStream(name);
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.reset();
			out.close();
			fileOut.close();
			System.out.println("Serialization Complete");
		} catch (IOException i){
			i.printStackTrace();
		}
	}

	/**
	 * Loads the serial file containing instance of DN2.
	 *
	 * @param name Name of the serial file
	 * @return DN2 Object instance
	 * @throws ClassNotFoundException This exception is thrown when there is no DN2 class in the project
	 */
	public DN2 deserializeLoad(String name) throws ClassNotFoundException{
		DN2 new_net = null;
		try{
			//FileInputStream fileIn = new FileInputStream("Image_Data/" + name + ".ser");
			FileInputStream fileIn = new FileInputStream(name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			new_net = (DN2)in.readObject();
			System.out.println("Deserialization Successful");
			in.close();
			fileIn.close();
		}catch(IOException i) {
			i.printStackTrace();
			return new_net;
		}
		return new_net;
	}


	/**
	 * This method returns the number of SensorLayers used by the network.
	 *
	 * @return numSensor Number of Sensors
	 */
	public int getNumSensor() {
		return numSensor;
	}

	/**
	 * This method sets the number of SensorLayers.
	 *
	 * @param numSensor Update the number of sensors.
	 */
	public void setNumSensor(int numSensor) {
		this.numSensor = numSensor;
	}


	/**
	 * This method returns the number of MotorLayers used by the network.
	 *
	 * @return numMotor Number of Motors
	 */
	public int getNumMotor() {
		return numMotor;
	}

	/**
	 * This method sets the number of MotorLayers.
	 *
	 * @param numMotor Update the number of sensors.
	 */
	public void setNumMotor(int numMotor) {
		this.numMotor = numMotor;
	}


	/**
	 * This method returns the number of HiddenLayers used by the network.
	 *
	 * @return numHidden Number of Motors
	 */
	public int getNumHidden() {
		return numHidden;
	}

	/**
	 * This method sets the number of HiddenLayers.
	 *
	 * @param numHidden Update the number of Y areas.
	 */
	public void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}


	// Need to figure how to return all these arrays by value, not by reference

	/**
	 * This method returns the array of SensorLayer instances used by the network.
	 *
	 * @return sensor Array of SensorLayer instances
	 */
	public SensorLayer[] getSensor() {
		return sensor;
	}

	/**
	 * This method sets an array of SensorLayer instances to the DN2 network.
	 *
	 * @param sensor Input array of SensorLayer instances
	 */
	public void setSensor(SensorLayer[] sensor) {
		this.sensor = sensor;
	}


	/**
	 * This method returns the array of MotorLayer instances used by the network.
	 *
	 * @return motor Array of MotorLayer instances
	 */
	public MotorLayer[] getMotor() {
		return motor;
	}

	/**
	 * This method sets an array of MotorLayer instances to the DN2 network.
	 *
	 * @param motor Input array of MotorLayer instances
	 */
	public void setMotor(MotorLayer[] motor) {
		this.motor = motor;
	}


	/**
	 * This method returns the array of HiddenLayer instances used by the network.
	 *
	 * @return hidden Array of HiddenLayer instances
	 */
	public HiddenLayer[] getHidden() {
		return hidden;
	}

	/**
	 * This method sets an array of HiddenLayer instances to the DN2 network.
	 *
	 * @param hidden Input array of HiddenLayer instances
	 */
	public void setHidden(HiddenLayer[] hidden) {
		this.hidden = hidden;
	}

	/**
	 * Update the 3D Space location of the Y neurons.
	 */
	public void updateHiddenLocation(){
		float pullrate = 0.1f;
		for(int i = 0; i < numHidden; i++){
			hidden[i].pullneurons(pullrate);
			hidden[i].calcuateNeiDistance();
		}
	}

	/**
	 * This method writes 3D space locations for Glial cells and Hidden neurons.
	 */
	public void outputHiddenLocation(){
		String filename = "Speech_Data/hidden_location.txt";
		try {
			PrintWriter wr = new PrintWriter(new File(filename));

			wr.println("Glial cells Location");

			for (int i = 0; i < numHidden; i++) {
				for (int j = 0; j < hidden[i].numGlial; j++) {
					int index = hidden[i].glialcells[j].getindex();
					float[] temp1 = hidden[i].glialcells[j].getlocation();
					//wr.println("Glialcell "+index+"'s location: ("+temp1[0]+", "+temp1[1]+", "+temp1[2]+")");
					wr.println(temp1[0]+", "+temp1[1]+", "+temp1[2]);
				}
				wr.println();
			}
			wr.println("Hidden Neourons Location");
			for (int i = 0; i < numHidden; i++) {
				wr.println("Hidden area: " + i);
				for (int j = 0; j < numHiddenNeurons[i]; j++) {
					int index = hidden[i].hiddenNeurons[j].getindex();
					float[] temp2 = hidden[i].hiddenNeurons[j].getlocation();
					//wr.println("Neuron "+index+"'s location: ("+temp2[0]+", "+temp2[1]+", "+temp2[2]+")");
					wr.println(temp2[0]+" "+temp2[1]+" "+temp2[2]);
				}

				wr.println();
			}
			wr.close();
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	/**
	 * This method sets which motor neurons will have topk competitions.
	 *
	 * True = Motor Neurons compute z responses for Hidden Neuron weight updates
	 * False = Motor Neurons lateral connections
	 *
	 * @param isTopk Determine whether the motor is used for Y or lateral Z updates
	 */
	public void setMotorisTopk(boolean isTopk){
		for(int i=0; i<motor.length; i++){
			motor[i].setisTopk(isTopk);
		}
	}

	/**
	 * This method sets which motor will follow mode?
	 *
	 * @param index The motor to be set for competition
	 */
	public void setMotorCompetition(int index){
		motor[index].setMode(true);
	}

	/**
	 *
	 * @param d
	 */
	public void set5grow(boolean d){
		for(int i=0; i<hidden.length; i++){
			hidden[i].set5growthrate(d);
		}
	}

	/**
	 * This method sets which neurons are related to concepts across all hidden neurons.
	 *
	 * @param c3
	 */
	public void setConcept(boolean c3){
		for(int i=0; i<hidden.length; i++){
			hidden[i].setConcept(c3);
		}
	}

	/**
	 * This method sets which neurons need to update their weights (learn) across all hidden neurons.
	 *
	 * @param l Indicate whether this method was called during the learning stage
	 */
	public void setLearning(boolean l,int index){
		for(int i=0; i<hidden.length; i++){
			if(index == 5){
				hidden[i].set5Learning(l);
			}
			if(index == 4){
				hidden[i].set4Learning(l);
			}
		}
	}

//	/**
//	 * This method returns the 3D Space position for each Y neuron
//	 *
//	 * @return 3D Array of each neuron's location.
//	 */
//	public float[][][] send_y_location(){
//		float[][][] temp = new float[numHidden][hidden[0].getUsedHiddenNeurons()][3];
//		for (int i = 0; i < numHidden; i++) {
//			for (int j = 0; j < hidden[i].getUsedHiddenNeurons(); j++) {
//				for(int k = 0; k < 3; k++){
//				 temp[i][j][k] = hidden[i].hiddenNeurons[j].getlocation()[k]/9.0f;}
//				}
//			}
//		return temp;
//	}

	/**
	 * This method returns the 3D Space position for each Z neuron
	 *
	 * @return 3D Array of each neuron's location.
	 */
	public float[][][] send_z_location(){
		float[][][] temp = new float[numMotor][][];
		for (int i = 0; i < numMotor; i++){
			temp[i] = new float[motor[i].getNumNeurons()][3];
		}
		for (int i = 0; i < numMotor; i++) {
			for (int j = 0; j < motor[i].getNumNeurons(); j++) {
				for(int k = 0; k < 3; k++){
					temp[i][j][k] = Math.min((motor[i].motorNeurons[j].getlocation()[k]+5.0f),12.0f)/15.0f;}
			}
		}
		return temp;
	}

	/**
	 * This method returns the y bottomup weights from a range of hidden neurons.
	 *
	 * @param num Selected number of neurons to visualize the bottomup weights of hidden neurons
	 * @param start_id The first neuron in the range
	 * @return The range of y bottomup weights
	 */
	public float[][] send_y_bottomup_weights(int num, int start_id){

		float[][]temp = hidden[0].send_y(num, start_id, 1);
		normalize(temp);
		return temp;
	}

	/**
	 * This method returns the y topdown weights from a range of hidden neurons.
	 *
	 * @param num Selected number of neurons to visualize the topdown weights of hidden neurons
	 * @param start_id The first neuron in the range
	 * @return The range of y topdown weights
	 */
	public float[][] send_y_topdown_weights(int num, int start_id){

		float[][] temp = hidden[0].send_y(num, start_id, 5);
		normalize(temp);
		return temp;
	}

	/**
	 * This method returns the y lateral weights from a range of hidden neurons.
	 *
	 * @param num Selected number of neurons to visualize the lateral weights of hidden neurons
	 * @param start_id The first neuron in the range
	 * @return The range of y lateral weights
	 */
	public float[][] send_y_lateral_weights(int num, int start_id){

		float[][] temp = hidden[0].send_y(num, start_id, 9);
		normalize(temp);
		return temp;
	}

	/**
	 *
	 * @param num
	 * @param start_id
	 * @return
	 */
	public float[][] send_y_lateral_masks(int num, int start_id){

		float[][] temp = hidden[0].send_y(num, start_id, 11);
		return temp;
	}

	/**
	 * This method returns the y inhibition weights from a range of inhibitory neurons.
	 *
	 * @param num Selected number of neurons to visualize the y inhibitory neurons
	 * @param start_id The first neuron in the range
	 * @return The range of y inhibitory weights
	 */
	public float[][] send_y_inhibition_weights(int num, int start_id){

		float[][] temp = hidden[0].send_y(num, start_id, 13);
		normalize(temp);
		return temp;
	}

	/**
	 * This method returns the y inhibition masks from a range of inhibitory neurons.
	 *
	 * @param num Selected number of neurons to visualize the inhibitory masks
	 * @param start_id The first neuron in the range
	 * @return The range of inhibitory masks
	 */
	public float[][] send_y_inhibition_masks(int num, int start_id){

		float[][] temp = hidden[0].send_y(num, start_id, 14);
//		normalize(temp);
		return temp;
	}

	/**
	 * This method returns the y responses from a range of hidden neurons.
	 *
	 * @param num Selected number of neurons to visualize the responses
	 * @param start_id The first neuron in the range
	 * @return The range of y responses
	 */
	public float[] send_y_response(int num, int start_id){
		//get the responses vector
		float[] temp = hidden[0].getResponse1D();
		//normalize to [0,1]
		normalize2(temp);
		return temp;
	}

	/**
	 * This method returns the bottomup weights of all motor neurons.
	 *
	 * @param start_id The first neuron in the range
	 * @return All the bottomup weights from [@start_id to end]
	 */
	public float[][] send_z_bottomup_weights(int start_id){
		int num = 0;
		//calculate the number of all z neurons
		for(int i=0; i<numMotor; i++){
			num += motor[i].getNumNeurons();
		}
		//initialize the weights matrix
		float[][] temp = new float[num][];

		int index=0;
		//copy weights from z neurons to weights matrix
		for(int i=0; i<numMotor; i++){
			System.arraycopy(motor[i].send_motor(motor[i].getNumNeurons(), 1), 0, temp, index , motor[i].getNumNeurons());
			index += motor[i].getNumNeurons();
		}
		for(int i = 0; i < num; i++){
			normalize2(temp[i]);
		}
		return temp;
	}

	/**
	 * This method returns the bottomup weights from a range of motor neurons.
	 *
	 * @param num Selected number of neurons to visualize the weights
	 * @param start_id The first neuron in the range
	 * @return The range of bottomup weights
	 */
	public float[][] send_z_bottomup_weights(int num, int start_id){

		float[][]temp = motor[0].send_motor(num, start_id);
		normalize(temp);
		return temp;
	}

//	/**
//	 * This method returns an array containing the number of neurons
//	 * of each HiddenLayer or Y area currently used by the DN2.
//	 *
//	 * @return int[] Array of used neurons per Y area.
//	 */
//	public int[] getUsedNeurons(){
//		int[] used_count = new int[numHidden];
//		for (int i = 0; i < numHidden; i++){
//			used_count[i] = hidden[i].getUsedHiddenNeurons();
//		}
//		return used_count;
//	}

	/**
	 * This method normalizes a 2D array of real numbers where each component has a value [0,1].
	 * It updates the values from the original vector.
	 *
	 * @param weight 2D array of float numbers
	 */
	public void normalize(float[][] weight){
		float min = weight[0][0];
		float max = weight[0][0];
		for (int i = 0; i < weight.length; i++){
			for (int j = 0; j < weight[0].length; j++){
				if(weight[i][j] < min){min = weight[i][j];}
				if(weight[i][j] > max){max = weight[i][j];}
			}
		}

		float diff = max-min + 0.0001f;
		for(int i = 0; i < weight.length; i++){
			for(int j = 0; j < weight[0].length; j++){
				weight[i][j] = (weight[i][j]-min)/diff;
			}
		}
	}

	/**
	 * This method normalizes an array of real numbers where each component has a value [0,1].
	 * It updates the values from the original vector.
	 *
	 * @param weight Array of float numbers
	 */
	public void normalize2(float[] weight){
		float min = weight[0];
		float max = weight[0];
		//find max and min value
		for (int i = 0; i < weight.length; i++){
			if(weight[i] < min){min = weight[i];}
			if(weight[i] > max){max = weight[i];}
		}
		//calculate the difference between max and min value
		float diff = max-min + 0.0001f;
		for(int i = 0; i < weight.length; i++){
			weight[i] = (weight[i]-min)/diff;
		}
	}

	/**
	 * This method returns the total number of connections per HiddenLayer.
	 *
	 * @return result An int array containing the number of connections per hidden layer.
	 */
	public int[] getConnections() {
		int[] result = new int[numHidden];
		for(int i=0; i<numHidden; i++) {
			result[i] = hidden[i].sumConnections();
		}
		return result;
	}

	public void setMotorConcept(int motorIndex, boolean isLearning) {
		motor[motorIndex].mConceptLearning = isLearning;
	}

	protected DN2() {}

	public void loadAllocations()
	{
	}

//	public void reinitializeRenderScript(Context context)
//	{
//	}

	public void finish() {}
	public void saveAllocations() {}

	protected void Setup(int[][] inputSize, int numSensor, int[][] motorSize, int numHidden, int[] numHiddenNeurons, int[][] rfSizes, int[][] rfStrides,
						 float lateralpercent,  MODE network_mode)
	{
		// Initialize the layers
		this.mode = network_mode;

		// calculate total sizes of the sensor
		int totalSensor = totalSize(inputSize);
		// calculate total sizes of the motor
		int totalMotor = totalSize(motorSize);
		//get the number of y neurons
		this.numHiddenNeurons = numHiddenNeurons;

		// Initialize the hidden layers
		//get the number of y layers
		this.numHidden = numHidden;
		//initialize the hidden-layer array
		hidden = new HiddenLayer[numHidden];
		//initialize each component of hidden-layer array
		this.lateralPercent = lateralpercent;

		//handle receptive field setup in RS
		this.rfSizes = rfSizes;
		this.rfStrides = rfStrides;

		//compute numColumns, numRows, numBoxes
		numBoxes = 0;
		numRows = 0;
		numColumns = 0;

		//compute numColumns, numRows, numBoxes
		numBoxes = 0;
		numRows = 0;
		numColumns = 0;

		int[] index = {0,0}; //x,y

		for (int i = 0; i < numSensor; i++){
			int width = rfSizes[i][1];
			int height = rfSizes[i][0];
			int stridex = rfStrides[i][1];
			int stridey = rfStrides[i][0];

			while (index[1] + stridey <= inputSize[i][0]){
				while (index[0] + stridex <= inputSize[i][1]){
					index[0] += stridex;
					numColumns++;
				}
				index[1] += stridey;
				numRows++;
			}
			index = new int[]{0,0};
			numBoxes = numBoxes + numRows * numColumns;
			numRows = 0;
			numColumns = 0;
		}

	}


	public float[] epsilon_normalize(float[] weight, int length, int column) {
		length = length - 2 -1; //- 2 reinforcers, - 1 volume dimension
		float[] new_weights = new float[weight.length];
		float norm = 0;
		float mean = 0;
		int useful_length = 0;
		for (int i = 0; i < length; i++) {
			//		if (mask[i] > 0) {
			mean += weight[i];
			useful_length++;

		}
		//calculate the mean value of non-zero elements in the vector
		mean = mean / useful_length - HiddenLayer.MACHINE_FLOAT_ZERO;

		//each non-zero element minuses the mean value
		for (int i = 0; i < length; i++) {
			//	if (mask[i] > 0) {
			weight[i] -= mean;
			//	}
		}
		//calculate the norm-2 for the non-zero elements
		for (int i = 0; i < length; i++) {
			//if (mask[i] > 0) {
			norm += weight[i] * weight[i];

		}
		norm = (float) Math.sqrt(norm) + HiddenLayer.MACHINE_FLOAT_ZERO;

		//normalize the non-zero element
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				weight[i] = weight[i] / norm;
			}
		}

		if(norm > maxVolumes[column]) {
			maxVolumes[column] = norm;
		}
		// TODO: Currently running: VD without volumeContribution
		//new_weights[weight.length-1-numMotor*2] is the volume dimension point, subtracting the reinforcers
		new_weights[weight.length-1-numMotor*2] = Neuron.volumeContribution * (maxVolumes[column] - norm) / maxVolumes[column];
		//weight[weight.length-1] = (maxVolume - norm) / maxVolume;
		norm = maxVolumes[column] / ((1-Neuron.volumeContribution) * norm);
		//norm = maxVolume / ( norm);
		//norm /= (1- volumeContribution);

		//if(length > 50)// && nor)
		//mNorm = norm;
		//normalize the non-zero element
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				new_weights[i] = weight[i] / norm;
			}
		}
		
		length = length + 2 +1; //+ 2 reinforcers, + 1 volume dimension

		norm = 0;
		for (int i = 0; i < length; i++) {
			//if (mask[i] > 0) {
			norm += new_weights[i] * new_weights[i];

		}
		norm = (float) Math.sqrt(norm) + HiddenLayer.MACHINE_FLOAT_ZERO;
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				new_weights[i] = new_weights[i] / norm;
			}
		}
		return new_weights;
	}

	public float[] epsilon_normalize(float[] weight, int length) {
		float[] new_weights = new float[weight.length + 1];
		float norm = 0;
		float mean = 0;
		int useful_length = 0;
		for (int i = 0; i < length; i++) {
			//		if (mask[i] > 0) {
			mean += weight[i];
			useful_length++;

		}
		//calculate the mean value of non-zero elements in the vector
		mean = mean / useful_length - HiddenLayer.MACHINE_FLOAT_ZERO;

		//each non-zero element minuses the mean value
		for (int i = 0; i < length; i++) {
			//	if (mask[i] > 0) {
			weight[i] -= mean;
			//	}
		}
		//calculate the norm-2 for the non-zero elements
		for (int i = 0; i < length; i++) {
			//if (mask[i] > 0) {
			norm += weight[i] * weight[i];

		}
		norm = (float) Math.sqrt(norm) + HiddenLayer.MACHINE_FLOAT_ZERO;

		//normalize the non-zero element
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				weight[i] = weight[i] / norm;
			}
		}

		return new_weights;
	}

	//added by jacob for getting hidden response
	public float[] getHiddenResponse (int i){
		return hidden[i].getNewResponse1D();
	}

	//added by jacob for Firing Neuron Age
	public float getFiringNeuronAge (int layer, int neuron){
		return hidden[layer].getFiringNeuronsAges(neuron);
	}

	//added by jacob for firing neuron type
	public int getFiringNeuronType (int layer, int neuron){
		return hidden[layer].getFiringNeuronType(neuron);
	}

	//added by jacob for top down weights
	public float[] getTopDownWeights (int layer, int neuron){
		return hidden[layer].getTopDownWeights(neuron);
	}

	//added by jacob for top down weights
	public float[] getLateralWeights (int layer, int neuron){
		return hidden[layer].getLateralWeights(neuron);
	}

	//added by jacob for bottom up weights
	public float[] getBottomUpWeights (int layer, int neuron){
		return hidden[layer].getBottomUpWeights(neuron);
	}

	public boolean[] getWinnerFlags(int layer) {
		return hidden[layer].getWinnerFlags();
	}

	//added by jacob for initialize neuron types
	static int[] initializeNeuronInfo = new int[]{0,0,0,0,0,0,0};
	public static void setInitializeNeuronInfo(int type, int used){
		initializeNeuronInfo[type - 1] = used;
	}

	public int[] getInitializeNeuronInfo(){
		return initializeNeuronInfo;
	}

	public ArrayList<float[][]> getMotorWeights()
	{
		ArrayList<float[][]> motorWeights = new ArrayList<>();
		for(int i = 0; i < motor.length; i++)
		{
			motorWeights.add(i, motor[i].getBottomUpWeights());
		}
		return motorWeights;
	}

	public float[][] getSensorGrid()
	{
		return sensorGrid;
	}

	public void setSerotoninMultiplier(float multiplier){
		hidden[0].setSerotoninMultiplier(multiplier);
		motor[0].setSerotoninMultiplier(multiplier);
	}

	public void setDopamineMultiplier(float multiplier){
		hidden[0].setDopamineMultiplier(multiplier);
		motor[0].setDopamineMultiplier(multiplier);
	}

	public ArrayList<float[]> getMotorPreResponses(){
		ArrayList<float[]> motorPreResponses = new ArrayList<>();

		for (int i = 0; i < numMotor; i ++){
			motorPreResponses.add(motor[i].getMotorPreResponse());
		}
		return motorPreResponses;
	}

	public int getNumberRfs(){
		return numBoxes;
	}

}

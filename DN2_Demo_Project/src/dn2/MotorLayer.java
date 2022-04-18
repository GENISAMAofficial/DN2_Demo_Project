package dn2;

// Remove because it is not being used //import java.util.Arrays;
//Remove because it is not being used //import java.util.Random;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * <h1>MotorLayer</h1>
 *
 * <p>
 * This class receives user motor input and prepares it for the DN2.
 *
 * This motors represents any action, context, or concept used for top-down attention of the real-world. 
 *
 *
 * </p>
 */
public class MotorLayer implements Serializable {

	/**
	 * Class serial ID.
	 *
	 * Identifies the array of bytes of the object sent across network sockets or text files.
	 */
	private static final long serialVersionUID = 1L;


	private int mMotorIndex;


	/** The height of the motor concept zone */
	private int height;

	/** The width of the motor concept zone */
	private int width;

	/** The normalize frequency of bottom-up weight vector */
	private int bottomupFrequency;


	/** The motor input vector */
	private float[][] input;

	/** The number of winning neurons */
	private int topK;

	/** Current number of initialized motor neurons */
	private int usedMotorNeurons;

	/** The number of z neurons in the motor concept zone */
	private int numNeurons;

	/** The length of bottom-up weight vector */
	private int numBottomUpWeights;

	/** The length of lateral weight vector */
	private int numLateralWeights;

	/** Whether do top-k competition */
	private boolean isTopk;

	// /** */
	//private final float GAMMA = 2000;

	/** The machine zero value */
	private final float MACHINE_FLOAT_ZERO = 0.00001f;

	/** The perfect match value */
	private final float TOP_MOTOR_RESPONSE = 1.0f - MACHINE_FLOAT_ZERO;
	// private final float TOP_MOTOR_RESPONSE = 0.5f;

	/** The z neuron array */
	public Neuron[] motorNeurons;

	/** Task indicator value */
	private boolean mode;

	// Decides which process to use when calculating responses
	public DNHandler.RESPONSE_CALCULATION_TYPE responseCalculationType = DNHandler.RESPONSE_CALCULATION_TYPE.GOODNESS_OF_MATCH;

	// Decides if this concept zone is currently computing and learning or not
	public boolean mConceptLearning = true;

	/**
	 * Constructor for the MotorLayer.
	 *
	 * This constructor initializes the MotorLayer neurons will all its parameters. 
	 * No input is sent here.
	 *
	 * @param height The height dimension of the motor input
	 * @param width The width dimension of the motor input
	 * @param topK The number of winning neurons to be updated per learning frames
	 * @param hiddenSize The number of Y neuron in all HiddenLayer
	 * @param lateralSize The number of lateral Z motor neurons
	 */
	// some motor neurons may have lateral connections.
	public MotorLayer(int height, int width, int topK, int hiddenSize, int lateralSize, DNHandler.RESPONSE_CALCULATION_TYPE _responseCalculationType){

		//set response type
		responseCalculationType = _responseCalculationType;

		// Why is this set to False?
		// TODO: AK: Figure this out.
		mode = true; //false;
		//set the width of the motor concept zone
		this.setWidth(width);
		//set the height of the motor concept zone
		this.setHeight(height);
		//set the number of winner
		this.setTopK(topK);
		//set not use top-k competition
		isTopk = false;
		//construct the input vector
		input = new float[height][width];
		usedMotorNeurons = topK+39;
		//calculate the number of z neurons in the motor area
		numNeurons = height * width;
		//set the length of bottom-up weight vector
		numBottomUpWeights = hiddenSize;
		//set the length of lateral weight vector
		numLateralWeights = lateralSize;
		//construct the motor neuron array
		motorNeurons = new Neuron[numNeurons];
		//initialize the motor neurons and their locations
		for(int i=0; i<numNeurons; i++){
			float[] temp = new float[3];
			temp[0] = (float)(Math.random()*10);
			temp[1] = (float)(Math.random()*10);
			temp[2] = (float)(Math.random()*10);
			motorNeurons[i] = new Neuron(numBottomUpWeights,0, lateralSize,false,i);
			motorNeurons[i].setlocation(temp);
		}
	}

	//construct the motor layer
	/**
	 * Constructor for the MotorLayer.
	 *
	 * This constructor sets the motor input dimensions and the receives the current input.
	 *
	 * @param width The width dimension of the motor input
	 * @param height The height dimension of the motor input
	 * @param input The motor input vector
	 */
	public MotorLayer(int width, int height, float[][] input){
		mode = false;
		//set the width of the motor concept zone
		this.setWidth(width);
		//set the height of the motor concept zone
		this.setHeight(height);
		//set input vector
		this.setInput(input);
	}

	//hebbian learning for the z neurons and update winner neurons' bottom-up weight vector
	public void hebbianLearnMotor(float[] hiddenResponse){
		for (int i = 0; i < numNeurons; i++) {
			//if the neuron's response lagers than or equals to the perfect match value, update its weight
			/*if(motorNeurons[i].getnewresponse() >= TOP_MOTOR_RESPONSE){
				motorNeurons[i].hebbianLearnHidden(hiddenResponse);
			}*/
			// IF the neuron has a nonzero response, update its weights
			if (motorNeurons[i].getnewresponse() >= MACHINE_FLOAT_ZERO && mConceptLearning) {
				motorNeurons[i].hebbianLearnHidden(hiddenResponse);
				
			}
		}
	}

	/**
	 * Hebbian learning for the Z neurons and update winner neurons' bottom-up weight vector.
	 *
	 * The motor neuron will proceed to update its weights only if each neuron's response is larger than perfect match.
	 *
	 */
	public void hebbianLearnLateral(float[] lateralResponse, int firing_neuron){
		motorNeurons[firing_neuron].hebbianLearnLateral(lateralResponse);
	}

	/**
	 * This method converts the motor neuron responses into a 1D Array.
	 *
	 * @return 1D Array of motor responses
	 */
	public float[] getNewMotorResponse1D() {
		//construct the new 1-D array
		float[] inputArray = new float[height * width];
		//copy values to the 1-D array
		for(int i = 0; i < numNeurons; i++){
			inputArray[i] = motorNeurons[i].getnewresponse();
		}
		return inputArray;
	}


	/**
	 * This method returns the motor neuron responses as a 2D Array.
	 * These are the responses from the Y to Z' connections.
	 *
	 * @return 2D Array of motor responses
	 */
	public float[][] getNewMotorResponse2D() {
		//construct the new array
		float[][] outputArray = new float[height][width];
		//copy the response values to the array
		for (int i = 0; i < numNeurons; i++) {
			outputArray[i/width][i%width] = motorNeurons[i].getnewresponse();
		}
		return outputArray;
	}

	/**
	 * This method returns the lateral motor neuron responses as a 2D Array.
	 * These are the responses from Z to Z' connections.
	 *
	 * @return 2D Array of motor responses
	 */
	public float[][] getLateralResponse2D() {
		//construct the new array
		float[][] outputArray = new float[height][width];
		//copy the lateral response values to the array
		for (int i = 0; i < numNeurons; i++) {
			outputArray[i/width][i%width] = motorNeurons[i].getlateralresponse();
		}
		return outputArray;
	}

	public float[] getLateralWeights(int neuron) {
		return motorNeurons[neuron].getLateralWeights();
	}

	/**
	 * This method replaces the old motor response values at time t-1
	 * with the computed motor response values at time t.
	 */
	public void replaceMotorLayerResponse(){
		for (int i = 0; i < numNeurons; i++) {
			motorNeurons[i].replaceResponse();
		}
	}

	/**
	 * This method computes the response from Z to Z' or lateral connections.
	 *
	 * @param lateral_input Array of Lateral Z responses
	 */
	public void computeLateralResponse(float[] lateral_input) {
		//normalize the lateral input vector
		normalize(lateral_input, lateral_input.length, 2);
		for (int i = 0; i < numNeurons; i++) {
			motorNeurons[i].computeLateralResponse(lateral_input, 1);
		}
	}

	/**
	 * This method computes the response from Y to Z' connections.
	 *
	 * @param hiddenResponse Array of Y neuron responses
	 */
	public void computeResponse(float[] hiddenResponse){
		float pain = hiddenResponse[hiddenResponse.length-2];
		float sweet = hiddenResponse[hiddenResponse.length-1];
		//normalize the bottom-up input vector
		normalize(hiddenResponse, hiddenResponse.length, 2);
		// do the dot product between the weights
		for (int i = 0; i < numNeurons; i++) {
			// Get around normalization
			hiddenResponse[hiddenResponse.length-2] = pain;
			hiddenResponse[hiddenResponse.length-1] = sweet;
			// Dot product of Y response and Z bottom-up weights

			motorNeurons[i].computeBottomUpResponse(hiddenResponse);

			// Since Z only has bottom-up weights, assign to total response
			motorNeurons[i].computeResponse(true);
		}

//		if (motorNeurons[0].getnewresponse() != 0f){
//			System.out.println("Response:" + motorNeurons[0].getnewresponse() + "Serotonin:" + motorNeurons[0].getSerotoninLevel());;
//		}

//		System.out.print("start here ");
//		for (int i = 0; i < numNeurons; i++) {
//
//			System.out.print(i + ":" + motorNeurons[i].getnewresponse() + ", ");
//		}
//		System.out.println();
//		System.out.println("end here");

		//do top-k competition
		if(isTopk){

			//
			if(mode == true){
				truetopKCompetition();
			}
			else{
				topKCompetition();
			}
		}
	}


	/**
	 * Sort the topK elements to the beginning of the sort array where the index of the top
	 * elements are still in the pair.
	 *
	 * @param sortArray Array of pairs, where each pair is composed of (response_value, neuron_index)
	 * @param topK Number of topk winners to be selected.
	 */
	private static void topKSort(Pair[] sortArray, int topK){
		//find the element with max value and record its index
		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i];
			int maxIndex = i;
			for (int j = i+1; j < sortArray.length; j++) {
				// select temporary max
				if(sortArray[j].value > maxPair.value){
					maxPair = sortArray[j];
					maxIndex = j;
				}
			}

			if(maxPair.index != i){
				// store the value of pivot (top i) element
				Pair temp = sortArray[i];
				// replace with the maxPair object
				sortArray[i] = maxPair;
				// replace maxPair index elements with the pivot
				sortArray[maxIndex] = temp;
			}
		}
	}

	/**
	 * This method computes the global topk competition in the Y area.
	 */
	private void topKCompetition(){
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons];
		//copy the z neurons' preResponses to the new pair array and a new array
		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, motorNeurons[i].getnewresponse());
//          System.out.println("Motor responses before topK: " + newResponse[i]);
			motorNeurons[i].setnewresponse(0.0f);
		}

		// Sort the array of Pair objects by its response_value in non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);


		//System.out.println("Motor top1 value: " + sortArray[0].value);

		// Find the top1 element and set to one.
		/*
		int topIndex = sortArray[0].get_index();
		newResponse[topIndex] = 1.0f;
		System.out.println("TopIndex: " + topIndex + " , " + "newResponse value: " + newResponse[topIndex]);
		*/

		// binary conditioning for the topK neurons.
		int winnerIndex = 0;
		while(winnerIndex < topK){
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].index;
			//set new response value for winner neuron
			motorNeurons[topIndex].setnewresponse(1.0f);
			winnerIndex++;
		}
	}

	/**
	 * This method computes the ranked global topk competition in the Y area.
	 */

	float[] copyArray;

	private void truetopKCompetition(){

		int winnerIndex = 0;

		copyArray = new float[numNeurons];
		// Pair is an object that contains the (index,response_value) of each hidden neurons.
		Pair[] sortArray = new Pair[numNeurons];
		//copy the z neurons' preResponses to the new pair array and a new array
		for (int i = 0; i < numNeurons; i++) {
			sortArray[i] = new Pair(i, motorNeurons[i].getnewresponse());
			copyArray[i] = motorNeurons[i].getnewresponse();

//          System.out.println("Motor responses before topK: " + newResponse[i]);
			motorNeurons[i].setnewresponse(0.0f);
		}

		// Sort the array of Pair objects by its response_value in non-increasing order.
		// The index is in the Pair, goes with the response ranked.
		topKSort(sortArray, topK);

		if(sortArray[0].value < 0.9f && usedMotorNeurons < numNeurons){ // add one more neuron.
//			System.out.println("This is the new hidden neuron " + usedMotorNeurons);
			motorNeurons[usedMotorNeurons].setnewresponse(1.0f);
			usedMotorNeurons++;
			winnerIndex++;
		}

		// identify the ranks of topk winners
		float value_top1 =  sortArray[0].value;
		float value_topkplus1 = sortArray[topK].value;
		// binary conditioning for the topK neurons.
		while(winnerIndex < topK){
			// get the index of the top element.
			int topIndex = sortArray[winnerIndex].index;
			//set new response value for winner neuron
			float tempResponse;
			switch (responseCalculationType) {

				// Calculate response based on relative goodness of match values
				case GOODNESS_OF_MATCH:
					if(value_top1 > value_topkplus1 ){ //this if statement prevents division by zero.
						tempResponse = ( copyArray[topIndex] - value_topkplus1 ) / (value_top1 - value_topkplus1 );
					}
					else{
						tempResponse = 1;
					}
					break;

				// Calculate response based on winner position in topk ranking
				case TRIANGLE_RESPONSE:
					tempResponse = 1.0f - ((float)winnerIndex / topK);
					break;
				default:
					throw new IllegalStateException("Unexpected value: " + responseCalculationType);
			}

			motorNeurons[topIndex].setnewresponse(tempResponse);
			winnerIndex++;
		}
//		System.out.println("Used number of motor neuron: "+usedMotorNeurons);
		//System.out.println(motorNeurons[0].getnewresponse() + ", " + motorNeurons[1].getnewresponse() + ", " + motorNeurons[2].getnewresponse());
	}

	/**
	 * Top-k competition for lateral connection or hidden to motor connection.
	 *
	 * True = Y to Z' competition
	 * False = Z to Z' competition (lateral connections)
	 *
	 * @param topk Which type of competition this MotorLayer will make.
	 */
	public void setisTopk(boolean topk){
		isTopk = topk;
	}

	/**
	 * Getter for the height dimension of the motor concept zone
	 *
	 * @return height dimension of the motor concept zone
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * Setter for the height dimension of the motor concept zone
	 *
	 * @param height The height dimension of the motor concept zone
	 */
	public void setHeight(int height) {
		this.height = height;
	}

	/**
	 * Getter for the width dimension of the motor concept zone
	 *
	 * @return width dimension of the motor concept zone
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * Setter for the width dimension of the motor concept zone
	 *
	 * @param width dimension of the motor concept zone
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * Convert the motor concept zone input into a 1D array.
	 *
	 * @return 1D input array
	 */
	public float[] getInput1D() {
		//construct the new 1-D array
		float[] inputArray = new float[height * width];
		//copy the input value to the 1-D array
		for (int i = 0; i < height; i++) {
			System.arraycopy(input[i], 0, inputArray, i * width, width);
		}
		return inputArray;
	}

	/**
	 * Setter for the topk competition mode.
	 *
	 * @param Mode Boolean value.
	 */
	public void setMode(boolean Mode) {
		this.mode = Mode;
	}

	/**
	 * Getter for the 2D motor concept input.
	 *
	 * @return 2D array of motor concept
	 */
	public float[][] getInput() {
		return input;
	}

	/**
	 * Setter for the 2D motor concept input.
	 *
	 * @param input 2D array of motor concept
	 */
	public void setInput(float[][] input) {
		this.input = input;
	}

	/**
	 * Getter for the number of winning neurons.
	 *
	 * @return The number of winning neurons in the topk competition.
	 */
	public int getTopK() {
		return topK;
	}

	/**
	 * Setter for the number of winning neurons.
	 *
	 * @param topK The number of winning neurons in the topk competition.
	 */
	public void setTopK(int topK) {
		this.topK = topK;
	}

	/**
	 * Setter for the index of the current MotorLayer instance.
	 * @param index
	 */
	public void setMotorindex(int index){
		mMotorIndex = index;
	}

	/**
	 * Getter for the number of z neurons in the motor concept zone.
	 *
	 * @return The number of used neurons by the MotorLayer
	 */
	public int getNumNeurons() {
		return numNeurons;
	}

	/**
	 * Getter for the number of bottom-up weights.
	 *
	 * @return The number of bottom-up weights
	 */
	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	/**
	 * Setter for the number of bottom-up weights.
	 *
	 * @param numBottomUpWeights The number of bottom-up weights
	 */
	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	/**
	 * Set the supervised motor responses.
	 *
	 * @param supervisedResponse 2D array of the motor input
	 */
	public void setSupervisedResponse(float[][] supervisedResponse){
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//calculate the index of neuron
				int index = i*width + j;
				motorNeurons[index].setnewresponse(supervisedResponse[i][j]);
				if (supervisedResponse[i][j] != 0) {
					motorNeurons[index].setConnected(true);
				}
			}
		}

	}

	public float[] getMotorResponse() {
		float[] tmp = new float[numNeurons];
		for (int i=0; i < numNeurons; i++) {
			tmp[i] = motorNeurons[i].getnewresponse();
		}
		return tmp;
	}

	public float[] getOldMotorResponse() {

		float[] tmp = new float[numNeurons];
		for (int i=0; i < numNeurons; i++) {
			tmp[i] = motorNeurons[i].getoldresponse();
		}
		return tmp;
	}


	/**
	 * This method normalizes a 1D array.
	 *
	 *  Flag values:
	 *
	 *  	1) Zero-mean epsilon norm
	 *  	2) L2-Norm (Euclidean Distance)
	 *  	3) Total-Sum Normalization
	 *
	 * @param input 1D array to be normalized
	 * @param size Number of elements in the input array
	 * @param flag Determines which normalization operation to use.
	 *
	 * @return Normalized 1D array
	 */
	public float[] normalize(float[] input, int size, int flag) {
		float[] weight = new float[size];
		System.arraycopy(input, 0, weight, 0, size);
		if (flag ==1){
			float min = weight[0];
			float max = weight[0];
			for (int i = 0; i < size; i++){
				if(weight[i] < min){min = weight[i];}
				if(weight[i] > max){max = weight[i];}
			}

			float diff = max-min + MACHINE_FLOAT_ZERO;
			for(int i = 0; i < size; i++){
				weight[i] = (weight[i]-min)/diff;
			}

			float mean = 0;
			for (int i = 0; i < size; i++){
				mean += weight[i];
			}
			mean = mean/size;
			for (int i = 0; i < size; i++){
				weight[i] = weight[i]-mean + MACHINE_FLOAT_ZERO;
			}

			float norm = 0;
			for (int i = 0; i < size; i++){
				norm += weight[i]*weight[i];
			}
			norm = (float) Math.sqrt(norm);
			if (norm > 0){
				for (int i = 0; i < size; i++){
					weight[i] = weight[i]/norm;
				}
			}
		}

		if(flag==2){
			float norm = 0;
			for (int i = 0; i < size; i++){
				norm += weight[i]* weight[i];
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			if (norm > 0){
				for (int i = 0; i < size; i++){
					weight[i] = weight[i]/norm;
				}
			}
		}

		if (flag == 3){
			float norm = 0;
			for (int i = 0; i < size; i++){
				norm += weight[i];
			}
			norm = norm+MACHINE_FLOAT_ZERO;
			if (norm > 0){
				for (int i = 0; i < size; i++){
					weight[i] = weight[i]/norm;
				}
			}
		}
		return weight;
	}

	/**
	 * Save the weight vectors into a text file.
	 *
	 * @param motor_ind Filename of the weights file.
	 */
	public void saveWeightToFile(String motor_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(motor_ind + "bottom_up_weight.txt"));
			PrintWriter wr_age = new PrintWriter(new File(motor_ind + "neuron_age.txt"));
			for (int i = 0; i < numNeurons; i++){
				for (int j = 0; j < motorNeurons[i].getBottomUpWeights().length; j++){
					wr_weight.print(String.format("% 2.2f", motorNeurons[i].getBottomUpWeights()[j]) + ',');
				}
				wr_weight.println();
				wr_age.print(Float.toString(motorNeurons[i].getfiringage()) + ',');
			}
			wr_weight.close();
			wr_age.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	/* This is the protocol for toy data visualization.
	 * All data are transfered as float to save effort in translation.
	 * There are these things to send over socket:
	 *       1. number of neurons in this motor
	 *       2. length of bottom up weights
	 *       3. bottom up weight
	 */
	/**
	 * This method will send motor inputs from the MotorLayer to a network socket.
	 *
	 * @param string_out
	 * @param data_out
	 * @param display_num Number of motor neurons requested
	 * @param display_start_id Index of the first motor neuron
	 *
	 * @throws IOException
	 */
	public void sendNetworkOverSocket(PrintWriter string_out, DataOutputStream data_out, int display_num, int display_start_id) throws IOException {
		int start_id = display_start_id - 1 ;
		if (start_id < 0) start_id = 0;
		if (start_id >= numBottomUpWeights) start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numBottomUpWeights) end_id = numBottomUpWeights;
		if (end_id < 0) end_id = numBottomUpWeights;

		data_out.writeInt(numNeurons);
		data_out.writeInt(end_id - start_id);
		data_out.writeInt(numLateralWeights);
		for(int i = 0; i < numNeurons; i++){
			for (int j = start_id; j < end_id; j++){
				data_out.writeFloat(motorNeurons[i].getBottomUpWeights()[j]);
			}
		}

		for (int i = 0; i < numNeurons; i++){
			for (int j = 0; j < numLateralWeights; j++){
				data_out.writeFloat(motorNeurons[i].getLateralWeights()[j]);
			}
		}
	}

	/**
	 * Send the weight and response vectors to the GUI.
	 *
	 * @param display_num Number of motor neurons requested
	 * @param display_start_id Index of the first motor neuron
	 *
	 * @return Current motor response 2D array
	 */
	public float[][] send_motor(int display_num, int display_start_id){
		float[][] temp_weights;
		int start_id = display_start_id - 1 ;
		if (start_id < 0) start_id = 0;
		if (start_id >= numBottomUpWeights) start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numBottomUpWeights) end_id = numBottomUpWeights;
		if (end_id < 0) end_id = numBottomUpWeights;
		temp_weights = new float[display_num][numBottomUpWeights];
		for(int i = 0; i < numNeurons; i++){
			for (int j = start_id; j < end_id; j++){
				temp_weights[i][j] = motorNeurons[i].getBottomUpWeights()[j];
			}
		}
		return temp_weights;
	}

	/**
	 * This class is used to sort the neurons for the topk competition.
	 * An instance of this class stores the pre-response value and index of a neuron.
	 */
	public class Pair implements Comparable<Pair> {

		/** Index of the neuron */
		public final int index;

		/** Pre-response value of the neuron */
		public final float value;

		/**
		 * This constructor sets the pre-response value and index of the neuron.
		 *
		 * @param index Neuron index
		 * @param value Pre-response value
		 */
		public Pair(int index, float value) {
			this.index = index;
			this.value = value;
		}

		/**
		 * This method is used to compare the pre-response values of the neurons when sorting.
		 */
		@Override
		public int compareTo(Pair other) {
			return -1*Float.valueOf(this.value).compareTo(other.value);
		}


	}

	public float[][] getBottomUpWeights()
	{
		float[][] weights = new float[numNeurons][numBottomUpWeights];
		for(int i = 0; i < numNeurons; i++)
		{
			weights[i] = motorNeurons[i].getBottomUpWeights();
		}
		return weights;
	}

	public void setSerotoninMultiplier(float multiplier){
		for (Neuron motorNeuron : motorNeurons) {
			motorNeuron.setSerotoninMultiplier(multiplier);
		}
	}

	public void setDopamineMultiplier(float multiplier){
		for (Neuron motorNeuron : motorNeurons) {
			motorNeuron.setDopamineMultiplier(multiplier);
		}
	}

	public float[] getMotorPreResponse(){
		return copyArray;
	}

}

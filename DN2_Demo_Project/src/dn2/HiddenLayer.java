package dn2;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

//import MazeInterface.Agent;
//import MazeInterface.Commons;
//import MazeInterface.DNCaller;

public class HiddenLayer implements Serializable {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;
	public int addNum;
	//the number of winner neurons
	protected int topK;
	//the number of neighbor neurons being pulled
	private int glialtopK;
	//the number of Y (hidden) neurons
	protected int numNeurons;
	//the percentage of lateral preResponse takes
	protected float lateralPercent;
	//the preResponse vector
	protected float[] preResponse;
	//the input size vector
	protected int[] inputSize;
	//the number of bottom-up connections
	protected int numBottomUpWeights;
	//the number of top-down connections
	protected int numTopDownWeights;
	//the receiptive field size
	private int[][] rfSize;
	//the stride of receiptive field
	private int[][] rfStride;
	//the matrix recording receiptive field location
	private int[][] rf_id_loc;
	//	private int usedHiddenNeurons;
	//the vector recording the number of used neuron for each type
	protected int[] typeindex;
	//number of neuron types
	protected int numTypes;
	//the matrix recording each type neurons' index
	protected float[][] areaMask;
	// The is the number of Y neurons
	public int[] usedNeurons;
	//private int[] winnerIndexs;
	//the vector recording growth rate
	private float[][] mGrowthRate;
	private float[][] mNeuronGrowth;
	//the vector recording the coefficient of mean value for dynamic top-k
	protected float[][] mMeanValue;
	//machine zero value
	static final float MACHINE_FLOAT_ZERO = 0.0001f;
	//the perfect match value
	private final float ALMOST_PERFECT_MATCH_RESPONSE = 1.0f - 10 * MACHINE_FLOAT_ZERO;
	//whether use dynamic top-k competition
	private boolean dynamicInhibition;
	//the percentage of prescreening keep
	private float prescreenPercent;
	private boolean type3;
	protected boolean type4learn;
	protected boolean type5learn;
	private boolean type5g;
	//the y neuron array
	public Neuron[] hiddenNeurons;
	//the inhibitory neuron array
	//public InhibitoryNeuron[] inhibitoryNeurons;
	//the size of glial cells
	private int glialsize;
	//the number of glial cells
	public int numGlial;
	//the glial cell array
	public Glial[] glialcells;
	public boolean isPriLateral;
	private int PriLateralsize;
	private float[] priLateralvector;
	public int numNeuronPerColumn;
	public int numColumns;
	public int numRows;
	public int numBoxes;
	public int numMotor;

	public static int numUpdates = 0;

	// Decides which process to use when calculating responses
	public DNHandler.RESPONSE_CALCULATION_TYPE responseCalculationType = DNHandler.RESPONSE_CALCULATION_TYPE.TRIANGLE_RESPONSE;
	protected int mAge = 0;

	private boolean yAttend;

	int[]sensorSizeSeperated;

	// TODO: AK: Fix the neuron number setup here to work for topk, k>1
	//the construction of hidden layer
	public HiddenLayer(int initialNumNeurons, int sensorSize, int[] sensorSizeSeperated, int motorSize, int[][] rfSizes, int[][] rfStrides,
					   int[][] rf_id_loc, int[][] inputSize, float prescreenPercent, int[] typeNum, float[][] growthTable, float[][] meanTable,
					   boolean dynamicinhibition, float[][] neuronGrowth, DNHandler.RESPONSE_CALCULATION_TYPE _responseCalculationType, int numBoxes, int numMotor) {

		//set response type
		responseCalculationType = _responseCalculationType;

		this.numBoxes = numBoxes;
		this.numMotor = numMotor;
		this.sensorSizeSeperated = sensorSizeSeperated;

		// TODO: AK added this
		sensorSize = 0;
		for (int i = 0; i < numBoxes; i++){
			sensorSize += sensorSizeSeperated[i];
		}
//		sensorSize += DN2.numMotor*2;
		this.setTopK(topK);
		//winnerIndexs = new int [topK];
//		this.usedHiddenNeurons = topK + 1; // bound of used neurons.
		//get used neuron types
		typeindex = new int[typeNum.length];
		for(int i=0; i<typeNum.length; i++){
			typeindex[i] = typeNum[i];
		}
		for(int i = 0; i < typeindex.length; i++){
			numTypes += typeindex[i]/(topK + 1);
		}

		this.addNum = 1;
		//initialize the number of used neurons
		this.usedNeurons = new int[numBoxes];

		//TODO: AK: Figure out if this needs to be topk+1 * numTypes not 2*numTypes (Made this change above and below)
		//TODO: AK: Or if it should just check that there are enough neurons and be initialNumNeurons only
		//initialize the number of total neurons
		this.numNeuronPerColumn = initialNumNeurons; // AK Changed this so that numneurons specified is the exact number of neurons//  + (topK+1) * numTypes; //2*numTypes;
		this.numNeurons= numNeuronPerColumn*numBoxes;
		//whether use dynamic inhibition or not
		this.dynamicInhibition = dynamicinhibition;
		//size of local receptive field
		this.rfSize = rfSizes;
		//moving step of local receptive field
		this.rfStride = rfStrides;
		//array of generated local receptive fields
		this.rf_id_loc = rf_id_loc;
		//initialize the input size
		this.inputSize = inputSize[0];
		//get the grow rate table values
		this.mGrowthRate = new float[growthTable.length][];
		for(int i = 0; i < growthTable.length; i++){
			mGrowthRate[i] = new float[growthTable[i].length];
			System.arraycopy(growthTable[i], 0, mGrowthRate[i], 0, growthTable[i].length);
		}

		this.mNeuronGrowth = new float[neuronGrowth.length][];
		for(int i = 0; i < neuronGrowth.length; i++){
			mNeuronGrowth[i] = new float[neuronGrowth[i].length];
			System.arraycopy(neuronGrowth[i], 0, mNeuronGrowth[i], 0, neuronGrowth[i].length);
		}

		//get the mean value table values
		this.mMeanValue = new float[meanTable.length][];
		for(int i = 0; i < meanTable.length; i++){
			mMeanValue[i] = new float[meanTable[i].length];
			System.arraycopy(meanTable[i], 0, mMeanValue[i], 0, meanTable[i].length);
		}

		//construct the area mask array
		areaMask = new float[7][numNeurons];

		//initialize the neurons

		hiddenNeurons = new Neuron[numNeurons];
		//inhibitoryNeurons = new InhibitoryNeuron[numNeurons];

		// TODO: AK: Get rid of the magic numbers in this code.
		//initialize each initial neuron
		for (int cl=0;cl<numBoxes;cl++) { // For each of the 9 columns in 3dEye
			{
				int number = 0;
				int topkPlusOne = topK + 1;

				this.usedNeurons[cl] = topkPlusOne*numTypes;
				int bias = cl*numNeuronPerColumn; // neuron number if in a list

				// initializes two neurons per column of every type designated
				for(int i=0; i<typeindex.length; i++){
					if(typeindex[i]!=0){
						// TODO: AK: Make this topk+1 neurons (not just 2).
						for(int j=number*topkPlusOne;j<(number+1)*topkPlusOne;j++){
							// TODO: AK: sensorSize / 9
							hiddenNeurons[bias+j] = new Neuron(sensorSizeSeperated[cl],motorSize,numNeuronPerColumn, true,(i+1), j,this.inputSize, numMotor);
						}
						number++;
					}
				}

				// Creates the rest of the neurons for each column (type=0) -> initializes not-used neurons
				for(int i=topkPlusOne*numTypes;i<numNeuronPerColumn;i++){
					// TODO: AK: sensorSize / 9
					hiddenNeurons[bias+i] = new Neuron(sensorSizeSeperated[cl],motorSize,numNeuronPerColumn, true,0, i,this.inputSize, numMotor);
				}


				// Creates k+1 inhibitory neurons per column of every type designated
//				number = 0;
//				for(int i=0; i<typeindex.length; i++){
//					if(typeindex[i]!=0){
//				        for(int j=number*topkPlusOne;j<(number+1)*topkPlusOne;j++){
//							// TODO: AK: sensorSize / 9
//					        inhibitoryNeurons[bias+j] = new InhibitoryNeuron(sensorSize,motorSize,numNeuronPerColumn, true,(i+1), j,this.inputSize);
//				        }
//				        number++;
//					}
//				}

				//initialize the not-used inhibitory neurons
//				for(int i=topkPlusOne*numTypes;i<numNeuronPerColumn;i++){
//					// TODO: AK: sensorSize / 9
//					inhibitoryNeurons[bias+i] = new InhibitoryNeuron(sensorSize,motorSize,numNeuronPerColumn, true,0, i,this.inputSize);
//				}

				// TODO: AK: Set topk+1 locations correctly
				//set initial neurons' locations
				for(int i=0;i<numTypes;i++){
					float[] temp = {4.5f+(float)Math.pow(-1, i)*0.1f*(i/2), 4.5f, 4.5f};
					for(int j=0; j<topkPlusOne; j++) {
						hiddenNeurons[topkPlusOne*i + j].setlocation(temp);
						temp[1]+=0.1f;
					}
					/*hiddenNeurons[bias+2*i].setlocation(temp);
					temp[1]+=0.1f;
					hiddenNeurons[bias+2*i+1].setlocation(temp);*/

				}

				//set the initial inhibitory neurons' locations
//				for(int i=0;i<numTypes;i++){
//					float[] temp = {4.5f+(float)Math.pow(-1, i)*0.1f*(i/2), 4.5f, 4.6f};
//					for(int j=0; j<topkPlusOne; j++) {
//						inhibitoryNeurons[topkPlusOne*i + j].setlocation(temp);
//						temp[1]+=0.1f;
//					}
//					/*inhibitoryNeurons[bias+2*i].setlocation(temp);
//					temp[1]+=0.1f;
//					inhibitoryNeurons[bias+2*i+1].setlocation(temp);*/
//
//				}

//				// Each type neuron initial two neurons have global receptive fields.
//				for (int i = 0; i < numTypes; i++) {
//					for (int j = 0; j < sensorSize; j++) {
//						for(int k=i*topkPlusOne;k<(i+1)*topkPlusOne;k++){
//							hiddenNeurons[bias+k].setBottomUpMask(1, j);
//							//inhibitoryNeurons[bias+k].setBottomUpMask(1, j);
//						}
//						/*hiddenNeurons[bias+2*i].setBottomUpMask(1, j);
//						hiddenNeurons[bias+2*i+1].setBottomUpMask(1, j);
//						inhibitoryNeurons[bias+2*i].setBottomUpMask(1, j);
//						inhibitoryNeurons[bias+2*i+1].setBottomUpMask(1, j);*/
//					}
//				}

//				// topDownMask are ones at the beginning
//				for (int i = 0; i < numNeuronPerColumn; i++) {
//					for (int j = 0; j < motorSize; j++) {
//						hiddenNeurons[bias+i].setTopDownMask(1, j);
//						//inhibitoryNeurons[bias+i].setTopDownMask(1, j);
//					}
//				}

				//set neurons' lateral mask
		/*		for (int i = 0; i < numNeurons; i++) {
					for (int j = 0; j < numNeurons; j++) {
						hiddenNeurons[i].setLateralMask(1, j);
					}
				}	*/

//				//inhibitory neurons' lateral mask only contain the same type neurons.
//				for (int i = 0; i <numTypes; i++) {
//					// AK: Changed this to work for topk not just k=1
//					for(int j=i*topkPlusOne;j<(i+1)*topkPlusOne;j++){
//						for(int k=i*topkPlusOne;k<(i+1)*topkPlusOne;k++){
//							inhibitoryNeurons[bias+j].setLateralWeight(1, k);
//							inhibitoryNeurons[bias+j].setLateralMask(1, k);
//						}
//					}
//					/*inhibitoryNeurons[bias+2*i].setLateralWeight(1, 2*i);
//					inhibitoryNeurons[bias+2*i].setLateralWeight(1, 2*i+1);
//					inhibitoryNeurons[bias+2*i+1].setLateralWeight(1, 2*i);
//					inhibitoryNeurons[bias+2*i+1].setLateralWeight(1, 2*i+1);
//					inhibitoryNeurons[bias+2*i].setLateralMask(1, 2*i);
//					inhibitoryNeurons[bias+2*i].setLateralMask(1, 2*i+1);
//					inhibitoryNeurons[bias+2*i+1].setLateralMask(1, 2*i);
//					inhibitoryNeurons[bias+2*i+1].setLateralMask(1, 2*i+1);*/
//				}

				//set the percentage of lateral excitation responses
				this.lateralPercent = 1.0f;

				this.prescreenPercent = prescreenPercent;
				//set the pre-response values
				this.preResponse = new float[numNeurons];
				for(int i=0; i<numNeuronPerColumn; i++){
					preResponse[bias+i] = 0;
				}

				numBottomUpWeights = sensorSize;
				numTopDownWeights = motorSize;

				// TODO: AK: Fix the area mask so it is outside this loop on the masks and sets up topk+1 neurons with ones
				//construct the area mask array
				//areaMask = new float[7][numNeurons];
				//initialize the area mask
				int tempindex = 0;
				for(int i=0; i<7; i++){ // For each neuron type
					for(int j=0; j<numNeuronPerColumn; j++){
						areaMask[i][bias+j] = 0;
						if (typeindex[i] !=0 && j < topkPlusOne) {
							areaMask[i][bias+j] = 1.0f;
						}
					}
					/*if(typeindex[i] !=0){
						areaMask[i][bias+2*tempindex]=1.0f;
						areaMask[i][bias+2*tempindex+1]=1.0f;
						tempindex += 1;
					}*/
				}
				yAttend = false;
				type3 = true;
				type4learn = true;
				type5learn = true;
				type5g =true;

				//initialize the glial cells
				//set the number of neighbor neurons to be pulled
				glialtopK = 3;
				//set the size of glial cells
				glialsize = 10;
				numGlial = glialsize*glialsize*glialsize;
				glialcells = new Glial[numGlial];
				for(int i=0; i<glialsize;i++){
					for(int j=0; j<glialsize;j++){
						for(int k=0; k<glialsize;k++){
							float[] temp = {(float)k,(float)j,(float)i};
							glialcells[glialsize*glialsize*i+glialsize*j+k] = new Glial(glialtopK,glialsize*glialsize*i+glialsize*j+k);
							glialcells[glialsize*glialsize*i+glialsize*j+k].setlocation(temp);
						}
					}
				}



			}
		}



	}

	//set the growth rate
	public void setGrowthRate(float[][] growth_table){
		//construct the growth rate array
		this.mGrowthRate = new float[growth_table.length][];
		for(int i = 0; i < growth_table.length; i++){
			mGrowthRate[i] = new float[growth_table[i].length];
			System.arraycopy(growth_table[i], 0, mGrowthRate[i], 0, growth_table[i].length);
		}
	}

	//set the neuron growth rate
	public void setNeuronGrowthRate(float[][] neuronGrowth){
		this.mNeuronGrowth = new float[neuronGrowth.length][];
		for(int i = 0; i < neuronGrowth.length; i++){
			mNeuronGrowth[i] = new float[neuronGrowth[i].length];
			System.arraycopy(neuronGrowth[i], 0, mNeuronGrowth[i], 0, neuronGrowth[i].length);
		}
	}

	public void saveAgeToFile(String hidden_ind, String index) {
		try {
			PrintWriter wr_age = new PrintWriter(new File(hidden_ind + "firing age "+index+".txt"));
			for (int i = 0; i < numNeurons; i++) {
				wr_age.print(Float.toString(hiddenNeurons[i].getfiringage()));
				wr_age.println();
			}
			wr_age.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	//save weights to a txt file
	public void saveWeightToFile(String hidden_ind) {
		try {
			PrintWriter wr_weight = new PrintWriter(new File(hidden_ind + "bottom_up_weight.txt"));
			PrintWriter wr_mask = new PrintWriter(new File(hidden_ind + "bottom_up_mask.txt"));
			PrintWriter wr_topdown = new PrintWriter(new File(hidden_ind + "top_down_weight.txt"));
			PrintWriter wr_topdownMask = new PrintWriter(new File(hidden_ind + "top_down_mask.txt"));
			PrintWriter wr_topdownAge = new PrintWriter(new File(hidden_ind + "top_down_age.txt"));
			PrintWriter wr_topDownVar = new PrintWriter(new File(hidden_ind + "top_down_var.txt"));
			for (int i = 0; i < numNeurons; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					wr_weight.print(String.format("% .2f", hiddenNeurons[i].getBottomUpWeights()[j]) + ',');
					//wr_mask.print(Float.toString(hiddenNeurons[i].getBottomUpMask()[j]) + ',');
				}
				wr_weight.println();
				wr_mask.println();

				for (int j = 0; j < hiddenNeurons[i].getTopDownWeights().length; j++) {
					wr_topdown.print(String.format("% .2f", hiddenNeurons[i].getTopDownWeights()[j]) + ',');
				}
				wr_topdown.println();

//				for (int j = 0; j < hiddenNeurons[i].getTopDownMask().length; j++) {
//					wr_topdownMask.print(String.format("% .2f", hiddenNeurons[i].getTopDownMask()[j]) + ',');
//				}
				wr_topdownMask.println();

//				for (int j = 0; j <  hiddenNeurons[i].getBottomUpWeights().length; j++) {
//					wr_topdownAge.print(String.format("% 4d", hiddenNeurons[i].gettopdownage()[j]) + ',');
//				}
//				wr_topdownAge.println();

//				for (int j = 0; j <  hiddenNeurons[i].getTopDownVariances().length; j++) {
//					wr_topDownVar.print(String.format("% .2f", hiddenNeurons[i].getTopDownVariances()[j]) + ',');
//				}
				wr_topDownVar.println();
			}
			wr_weight.close();
			wr_mask.close();
			wr_topdown.close();
			wr_topdownMask.close();
			wr_topdownAge.close();
			wr_topDownVar.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


//	// Initialize receptive field for the ith neuron, according to the whereID.
//	// The center of the receptive field is located at rf_id_loc[i].
//	// size of the receptive field is rf_size.
//	public void initializeRfMask(int i, int whereID, DN2.MODE mode) {
//        //initialize the receipt field for toy problem
//		if (mode == DN2.MODE.GROUP) {
//			if (whereID >= 0) {
//				int half_rf_size_w = (rfSize[1] - 1) / 2;
//				int half_rf_size_h = (rfSize[0] - 1) / 2;
//				System.out.println("Where: " + whereID + " " + rf_id_loc.length);
//				int rf_begin_row = rf_id_loc[whereID][1] - half_rf_size_w;
//				int rf_end_row = rf_id_loc[whereID][1] + half_rf_size_w;
//				int rf_begin_col = rf_id_loc[whereID][0] - half_rf_size_h;
//				int rf_end_col = rf_id_loc[whereID][0] + half_rf_size_h;
//
//				System.out.println(rf_begin_row + " " + rf_end_row);
//				System.out.println(rf_begin_col + " " + rf_end_col);
//
//				// assert inputSize[0] * inputSize[1] ==
//				// hiddenNeurons[i].getBottomUpMask().length;
//				for (int row = rf_begin_row; row <= rf_end_row; row++) {
//                    //initialize the bottom-up mask vector
//					for (int col = rf_begin_col; col <= rf_end_col; col++) {
//						int current_idx = col * inputSize[1] + row;
//						hiddenNeurons[i].setBottomUpMask(1.0f, current_idx);
//					}
//				}
//				for (int pixel_ind = inputSize[0] * inputSize[1]; pixel_ind < hiddenNeurons[i]
//						.getBottomUpMask().length; pixel_ind++) {
//					hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
//				}
//			} else {
//				for (int pixel_ind = 0; pixel_ind < hiddenNeurons[i].getBottomUpMask().length; pixel_ind++) {
//					hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
//				}
//			}
//			System.out.println("Rf initialized: " + whereID);
//        //initialize the receipt field for maze problem
//		}
//		else if(mode == DN2.MODE.Speech){
//			for (int pixel_ind = 0; pixel_ind < hiddenNeurons[i].getBottomUpMask().length; pixel_ind++) {
//				hiddenNeurons[i].setBottomUpMask(1.0f, pixel_ind);
//			}
//		}
//		/*else if (mode == DN2.MODE.MAZE){
//			if (Commons.vision_2D_flag == false){
//				int rf_loc = DNCaller.curr_loc;
//				int rf_size = DNCaller.curr_scale;
//				float[] bottom_up_mask = new float[currentBottomUpInput.length];
//	            //initialize the bottom-up mask vector
//				if (rf_loc < 0){
//					for (int j = 0; j < 3 * Agent.vision_num; j++){
//						bottom_up_mask[j] = 1;
//					}
//				} else {
//					for (int j = rf_loc * 3; j < (rf_loc + rf_size) * 3; j++){
//						bottom_up_mask[j] = 1;
//					}
//				}
//				for (int j = 3 * Agent.vision_num; j < currentBottomUpInput.length; j++){
//					bottom_up_mask[j] = 1;
//				}
//
//				hiddenNeurons[i].setBottomUpMasks(bottom_up_mask);
//
//			} else {
//				int rf_loc = DNCaller.curr_loc;
//				int rf_size = DNCaller.curr_scale;
//				int rf_type = DNCaller.curr_type;
//				float[] bottom_up_mask = new float[currentBottomUpInput.length];
//				if (rf_loc < 0){
//					for (int j = 0; j < currentBottomUpInput.length; j++){
//						bottom_up_mask[j] = 1;
//					}
//				} else {
//					int vision_height = Agent.vision_num * 3/4;
//					int count = 0;
//					for (int j = 0; j < vision_height; j++){
//						for (int k = 0; k < Agent.vision_num; k++){
//							if (k >= rf_loc - 1 && k <= (rf_loc + rf_size)){
//								bottom_up_mask[count * 3] = 1;
//								bottom_up_mask[count * 3 + 1] = 1;
//								bottom_up_mask[count * 3 + 2] = 1;
//							}
//							count ++ ;
//						}
//					}
//				}
//
//				hiddenNeurons[i].setBottomUpMasks(bottom_up_mask);
//
//			}
//		}*/
//	}

	public void addFiringAges() {
		for(int i=0;i<numBoxes;i++)
			for(int j=0; j<usedNeurons[i]; j++){
				hiddenNeurons[numNeuronPerColumn*i+j].addFiringAge();
				//inhibitoryNeurons[numNeuronPerColumn*i+j].addFiringAge();
			}
	}

	//hebbian learning for y neurons
	public void hebbianLearnHidden(float[][] sensorGrid, float[] motorInput, float[][] responseGrid) {
		mAge++; // AK Added this to keep track of DN age for growth rate
		//int cellSize = (int)Math.sqrt(sensorInput.length / (6)) / 3; // 3 boxes per row and col
		for (int cl=0;cl<numBoxes;cl++) {

			// TODO: Arden made this change
			float[] currentInput = new float[numBottomUpWeights];
			//System.arraycopy(sensorInput, cl*numBottomUpWeights, currentInput,0, numBottomUpWeights);

			currentInput = sensorGrid[cl];

			//currentInput = epsilon_normalize(currentInput, currentInput.length);

			int bias = cl*numNeuronPerColumn;
			boolean learning = true;
			float[] tempResponse1 = new float[numNeuronPerColumn];
			//float[] tempResponse2 = new float[numNeuronPerColumn];
			//get the response from last frame
			for(int i=0;i<numNeuronPerColumn;i++){
				tempResponse1[i] = hiddenNeurons[numNeuronPerColumn*cl+i].getoldresponse();
				//tempResponse2[i] = preResponse[numNeuronPerColumn*cl+i];
				//			winnerIndex[i] = hiddenNeurons[i].getwinnerflag();
				//			System.out.println("neuron "+i+" winnerflag "+hiddenNeurons[i].getwinnerflag());
			}
			//tempResponse1 = Neuron.epsilon_normalize(tempResponse1);
			int tempIndex = (int)(((float)(usedNeurons[cl])/numNeuronPerColumn)/0.05f);
			if(usedNeurons[cl] == numNeuronPerColumn){
				tempIndex = tempIndex-1;
			}

			for(int j=0; j<usedNeurons[cl]; j++){
				if(hiddenNeurons[numNeuronPerColumn*cl+j].getType() == 5){
					learning  = type5learn;
				}
				else if(hiddenNeurons[numNeuronPerColumn*cl+j].getType() == 4){
					learning  = type4learn;
				}
				else{
					learning = true;
				}

				if(learning){ //TODO: AK made this change
					//update weights for each y neuron
					//hiddenNeurons[numNeuronPerColumn*cl+j].hebbianLearnHidden(sensorInput, motorInput, tempResponse1);
					hiddenNeurons[numNeuronPerColumn*cl+j].hebbianLearnHidden(currentInput, motorInput, tempResponse1);

					if(hiddenNeurons[j].getIsGrowLateral()) {
						int[] temp = hiddenNeurons[numNeuronPerColumn*cl+j].getLateralGrowList();
						growLateralConnection(numNeuronPerColumn*cl+j, temp);
						hiddenNeurons[numNeuronPerColumn*cl+j].setIsGrowLateral(false);
						hiddenNeurons[numNeuronPerColumn*cl+j].clearLateralGrowList();
						hiddenNeurons[numNeuronPerColumn*cl+j].clearLateralDeleteList();
					}
					//update weights for each inhibitory neuron
					float[][] tmpAreaMask = new float[7][numNeuronPerColumn];
					for(int k=0;k<7;k++) {
						for(int l=0;l<numNeuronPerColumn;l++) {
							tmpAreaMask[k][l] = areaMask[k][bias+l];
						}
					}

					// TODO: AK made this change
					//inhibitoryNeurons[numNeuronPerColumn*cl+j].hebbianLearnHidden(sensorInput, motorInput, tempResponse2,tmpAreaMask);
//				inhibitoryNeurons[numNeuronPerColumn*cl+j].hebbianLearnHidden(currentInput, motorInput, tempResponse2,tmpAreaMask);
//	       			    //update the inhibition area for each neuron
//	      	        float[] tempWeights = inhibitoryNeurons[numNeuronPerColumn*cl+j].getLateralWeights();
//	        			float meanWeight = mean(tempWeights,tmpAreaMask[inhibitoryNeurons[numNeuronPerColumn*cl+j].getType()-1 ])*mMeanValue[tempIndex][1];
//	        			     for(int k = 0; k < usedNeurons[cl]; k++){
//	        			    	 if((tempWeights[k]>=meanWeight) && (tmpAreaMask[inhibitoryNeurons[numNeuronPerColumn*cl+j].getType()-1 ][k]!=0)){
//	        			    		 inhibitoryNeurons[numNeuronPerColumn*cl+j].setLateralMask(1, k);
//	        			    	 }
//	        			    	 else{
//	        			    		 inhibitoryNeurons[numNeuronPerColumn*cl+j].setLateralMask(0, k);
//	        			    	 }
//	        			     }
				}
			}
		}
	}

//	   //hebbian learning for y neurons
//		public void hebbianLearnHiddenParallel(float[][] sInputs, float[] sensorInput, float[] motorInput) {
//			boolean learning = true;
//			float[] tempResponse1 = new float[numNeurons];
//			float[] tempResponse2 = new float[numNeurons];
//	        //get the response from last frame
//			for(int i=0;i<numNeurons;i++){
//				tempResponse1[i] = hiddenNeurons[i].getoldresponse();
//				tempResponse2[i] = preResponse[i];
////				winnerIndex[i] = hiddenNeurons[i].getwinnerflag();
////				System.out.println("neuron "+i+" winnerflag "+hiddenNeurons[i].getwinnerflag());
//			}
//	        int tempIndex = (int)(((float)usedNeurons/numNeurons)/0.05f);
//	        if(usedNeurons == numNeurons){
//	        	tempIndex = tempIndex-1;
//	        }
//	        for(int j=0; j<usedNeurons; j++){
//	        	if(hiddenNeurons[j].getType() == 5){
//	        		learning  = type5learn;
//	            }
//	        	else if(hiddenNeurons[j].getType() == 4){
//	        		learning  = type4learn;
//	        	}
//	        	else{
//	        		learning = true;
//	        	}
//
//	        	if(learning){
//	            //update weights for each y neuron
//	        	hiddenNeurons[j].hebbianLearnHiddenParallel(sInputs, motorInput, tempResponse1);
//
//	        	if(hiddenNeurons[j].getIsGrowLateral()) {
//	        		int[] temp = hiddenNeurons[j].getLateralGrowList();
//	        		growLateralConnection(j, temp);
//	        		hiddenNeurons[j].setIsGrowLateral(false);
//	        		hiddenNeurons[j].clearLateralGrowList();
//	        		hiddenNeurons[j].clearLateralDeleteList();
//	        	}
//
//	            //update weights for each inhibitory neuron
//	        	inhibitoryNeurons[j].hebbianLearnHidden(sensorInput, motorInput, tempResponse2,areaMask);
//	       			    //update the inhibition area for each neuron
//	      	        float[] tempWeights = inhibitoryNeurons[j].getLateralWeights();
//	        			float meanWeight = mean(tempWeights,areaMask[inhibitoryNeurons[j].getType()-1 ])*mMeanValue[tempIndex][1];
//	        			     for(int k = 0; k < usedNeurons; k++){
//	        			    	 if((tempWeights[k]>=meanWeight) && (areaMask[inhibitoryNeurons[j].getType()-1 ][k]!=0)){
//	        			    		 inhibitoryNeurons[j].setLateralMask(1, k);
//	        			    	 }
//	        			    	 else{
//	        			    		 inhibitoryNeurons[j].setLateralMask(0, k);
//	        			    	 }
//	        			     }
//	          }
//	        }
//
//		}

	// convert into 1d Array
	public float[] getResponse1D() {
		//construct the new array
//		for(int i=0;)
		float[] inputArray = new float[numNeurons];
		//copy values
		for(int cl=0;cl<numBoxes;cl++)
			for (int j = 0; j < usedNeurons[cl]; j++) {
				inputArray[numNeuronPerColumn*cl+j] = hiddenNeurons[numNeuronPerColumn*cl+j].getoldresponse();
			}

		return inputArray;
	}

	//compute the bottom-up preResponse
	public void computeBottomUpResponse(float[][] sensorInput, int[] sensorSize) {
		// TODO: Arden disabled this because different inputs for different neurons
		// Keep track of the sensor Input
		int beginIndex = 0;
		ArrayList<Float> responses = new ArrayList<>();

		/*for (int j = 0; j < sensorSize.length; j++) {
			System.arraycopy(sensorInput[j], 0, currentBottomUpInput, beginIndex, sensorSize[j]);
			beginIndex += sensorSize[j];
		}*/
		//copy the current bottom-up input
		//float[] currentInput = new float[numBottomUpWeights];

		//System.arraycopy(currentBottomUpInput, 0, currentInput,0, numBottomUpWeights);

		float[] currentInput = new float[numBottomUpWeights];
//		int cellSize = (int)Math.sqrt(sensorInput[0].length / (6)) / 3; // 3 boxes per row and col
		for(int cl=0;cl<numBoxes;cl++) {
//			int row = cl/3;
//			int col = cl%3;
//			for(int im_idx = 0;im_idx<2;im_idx+=1) {
//				for(int r=0;r<cellSize;r++) {
//					for(int c=0;c<cellSize;c++) {
//						for(int color = 0;color<3;color++) {
//							int inputIdx = im_idx*(3*3*cellSize*cellSize*3)+(row*cellSize+r)*(cellSize*3*3)+(col*cellSize+c)*3+color;
//							currentInput[im_idx*(cellSize*cellSize*3) + r*cellSize*3 + 3*c + color] = (float) ((int)sensorInput[0][inputIdx] & 0xFF);
//						}
//					}
//				}
//			}
			currentInput = sensorInput[cl];

			/*int sze = currentInput.length;
			ByteBuffer bf = ByteBuffer.allocate(4*sze / 3);
			for (int i = 0; i < sze; i += 3) {
				int R = (int) currentInput[i] << 24;
				int G = (int) currentInput[i + 1] << 16;
				int B = (int) currentInput[i + 2] << 8;
				int A = 255; //(int) sensorInput[i + 3];
				bf.putInt(A + R + G + B);
			}
			Bitmap bitmapSensor = Bitmap.createBitmap(45, 2 * 45, Bitmap.Config.ARGB_8888);
			bf.position(0);
			bitmapSensor.copyPixelsFromBuffer(bf);*/
			// TODO: Arden made this change
			//float[] currentInput = new float[numBottomUpWeights];
			//System.arraycopy(sensorInput[0], cl*numBottomUpWeights, currentInput,0, numBottomUpWeights);


			for (int i = 0; i < usedNeurons[cl]; i++) {
//	        	System.out.println("neuron index: "+numNeuronPerColumn+";"+cl+";"+i+";"+(numNeuronPerColumn*cl+i));
				responses.add(hiddenNeurons[numNeuronPerColumn * cl + i].computeBottomUpResponse(currentInput));
			}
		}

	}

//    //compute the bottom-up preResponse
//	public void computeBottomUpResponseInParallel(float[][] sensorInput, int[] sensorSize) {
//		float[][] currentInput = new float[sensorSize.length][];
//		// Keep track of the sensor Input
//		int beginIndex = 0;
//		for (int j = 0; j < sensorSize.length; j++) {
//			System.arraycopy(sensorInput[j], 0, currentBottomUpInput, beginIndex, sensorSize[j]);
//			beginIndex += sensorSize[j];
//
//			//copy the current bottom-up input
//			currentInput[j] = new float[sensorSize[j]];
//			System.arraycopy(sensorInput[j], 0, currentInput[j], 0, sensorSize[j]);
//		}
//		for(int i=0; i<usedNeurons; i++){
//			    hiddenNeurons[i].computeBottomUpResponseInParallel(currentInput);
//		}
//	}

	//compute the top-down preResponse
	public void computeTopDownResponse(float[] motorInput, int[] motorSize) {
		// Keep track of the top-down Input
//		int beginIndex = 0;
//		for (int j = 0; j < motorSize.length; j++) {
//			System.arraycopy(motorInput[j], 0, currentTopDownInput, beginIndex, motorSize[j]);
//			beginIndex += motorSize[j];
//		}
		ArrayList<Float> responses = new ArrayList<>();
		//copy the top-down input
		float[] currentInput = new float[numTopDownWeights];
		currentInput = motorInput;

		//System.arraycopy(currentTopDownInput, 0, currentInput, 0, numTopDownWeights);


		// If using SM.
		for(int cl=0;cl<numBoxes;cl++)
			for(int i=0; i<usedNeurons[cl]; i++){
				responses.add(hiddenNeurons[numNeuronPerColumn*cl+i].computeTopDownResponse(currentInput));
			}

	}

	//compute the lateral preResponse
	public void computeLateralResponse(float[][] response){
		if(isPriLateral) {
			float[] tempResponse = new float[PriLateralsize+numNeurons];
			System.arraycopy(priLateralvector, 0,tempResponse, 0, PriLateralsize);
			float[] tempResponse2 = this.getResponse1D();
			System.arraycopy(tempResponse2, 0,tempResponse, PriLateralsize, numNeurons);
//			tempResponse = this.preResponse;

			for(int cl=0;cl<numBoxes;cl++)
				for(int i=0;i<usedNeurons[cl]; i++){
					hiddenNeurons[numNeuronPerColumn*cl+i].computeLateralResponse(tempResponse,lateralPercent);

				}
		}
		else {
			ArrayList<Float> responses = new ArrayList<>();
			float[] tempResponse = new float[numNeurons];
			tempResponse = this.getResponse1D();
//			tempResponse = this.preResponse;

			for(int cl=0;cl<numBoxes;cl++)
				for(int i=0;i<usedNeurons[cl]; i++){
					responses.add(hiddenNeurons[numNeuronPerColumn*cl+i].computeLateralResponse(tempResponse,lateralPercent));

				}
		}
	}

	// TODO: AK: Check this function
	//compute the final response
	public void computeResponse(int whereId, boolean learn_flag, DN2.MODE mode) {
		prescreenResponse();

		for(int cl=0;cl<numBoxes;cl++) {
//			float[] tempArr = new float[usedNeurons[cl]];
			for(int i=0;i<usedNeurons[cl]; i++){
				float num = hiddenNeurons[numNeuronPerColumn*cl+i].computeResponse(false);
//				tempArr[i] = num;
			}
//			System.out.println("pre_resp: " + Arrays.toString(tempArr));

			// do the topKcompetition
			float temp = (float)(usedNeurons[cl])/numNeuronPerColumn;
			int index = 0;
			if (mode == DN2.MODE.GROUP || mode == DN2.MODE.Speech) {
				index = (int)(temp/0.02f);
			}
			if(mode == DN2.MODE.MAZE){
				index = (int)(temp/0.05f);
			}
			if(usedNeurons[cl] == numNeuronPerColumn){
				index = index-1;
			}
			/*
			int index2 = 0;
			if(type5g){
				index2 = 4;
			}*/

			//dynamic top-k competition for each type neurons

		}


		if(dynamicInhibition){
			if(typeindex[4]!=0 && type5g == true){
				dynamic3x3topKCompetition(whereId, learn_flag,5, mode);
			}
			if(typeindex[0]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,1, mode);
			}
			if(typeindex[3]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,4, mode);
			}
			if(typeindex[2]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,3, mode);
			}
			if(typeindex[5]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,6, mode);
			}
			if(typeindex[6]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,7, mode);
			}
			if(typeindex[1]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,2, mode);
			}
		}
		//global top-k competition for each type neurons
		else{
			if(typeindex[4]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,5, mode);
			}
			if(typeindex[0]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,1, mode);
			}
			if(typeindex[3]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,4, mode);
			}
			if(typeindex[1]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,2, mode);
			}
			if(typeindex[2]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,3, mode);
			}
			if(typeindex[5]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,6, mode);
			}
			if(typeindex[6]!=0){
				dynamic3x3topKCompetition(whereId, learn_flag,7, mode);
			}
		}
        /*if (hiddenNeurons[9].getwinnerflag()) {
			System.out.println("");
			int sze = currentBottomUpInput.length;
			ByteBuffer bf = ByteBuffer.allocate(sze);
			for (int i = 0; i < sze; i += 4) {
				int A = (int) currentBottomUpInput[i] << 24;
				int R = (int) currentBottomUpInput[i + 1] << 16;
				int G = (int) currentBottomUpInput[i + 2] << 8;
				int B = (int) currentBottomUpInput[i + 3];
				bf.putInt(A + R + G + B);
			}
			Bitmap bitmapSensor = Bitmap.createBitmap(60, 2 * 60, Bitmap.Config.ARGB_8888);
			bf.position(0);
			bitmapSensor.copyPixelsFromBuffer(bf);


			ByteBuffer bf2 = ByteBuffer.allocate(sze);
			for (int i = 0; i < sze; i += 4) {
				int A = (int) (hiddenNeurons[9].getBottomUpWeights()[i] * hiddenNeurons[9].mNorm + hiddenNeurons[9].mMean) << 24;
				int R = (int) (hiddenNeurons[9].getBottomUpWeights()[i + 1] * hiddenNeurons[9].mNorm + hiddenNeurons[9].mMean) << 16;
				int G = (int) (hiddenNeurons[9].getBottomUpWeights()[i + 2] * hiddenNeurons[9].mNorm + hiddenNeurons[9].mMean) << 8;
				int B = (int) (hiddenNeurons[9].getBottomUpWeights()[i + 3] * hiddenNeurons[9].mNorm + hiddenNeurons[9].mMean);
				bf2.putInt(A + R + G + B);
			}
			Bitmap bitmapWeights = Bitmap.createBitmap(60, 2 * 60, Bitmap.Config.ARGB_8888);
			bf2.position(0);
			bitmapWeights.copyPixelsFromBuffer(bf2);

			System.out.println("");
		}*/
	}

	public void computeResponse(int whereId, boolean learn_flag, DN2.MODE mode, int type) {
		prescreenResponse();

		for(int cl=0;cl<numBoxes;cl++) {
			for(int i=0;i<usedNeurons[cl]; i++){

				if(hiddenNeurons[numNeuronPerColumn*cl+i].getType() == type){
					hiddenNeurons[numNeuronPerColumn*cl+i].computeResponse(false);
				}
				else{
					hiddenNeurons[numNeuronPerColumn*cl+i].setnewresponse(0);
				}
			}


//	        // do the topKcompetition
//			float temp = (float)usedNeurons[cl]/numNeuronPerColumn;
//			int index = 0;
//			if (mode == DN2.MODE.GROUP || mode == DN2.MODE.Speech) {
//				index = (int)(temp/0.02f);
//			}
//			if(mode == DN2.MODE.MAZE){
//			    index = (int)(temp/0.05f);
//			}
//			if(usedNeurons[cl] == numNeuronPerColumn){
//				index = index-1;
//			}
			//dynamic top-k competition for each type neurons
			dynamic3x3topKCompetition(whereId, learn_flag, type, mode);
		}
	}

	//do prescreening
	private void prescreenResponse() {
		// Prescreen bottomUpResponse
		float[] tempArray = new float[numNeurons];
		for(int i = 0; i < numNeurons; i++){
			tempArray[i] = hiddenNeurons[i].getbottomUpresponse();
		}
		//sort the bottom-up preResponses
		Arrays.sort(tempArray);
		int cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
		if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
		float cutOffValue = tempArray[cutOffPos];
		for (int i = 0; i < numNeurons; i++){
			if(hiddenNeurons[i].getbottomUpresponse() < cutOffValue){
				hiddenNeurons[i].setbottomUpresponse(0);
			}
		}

		// Prescreen topDownResponse
		tempArray = new float[numNeurons];
		for (int i = 0; i < numNeurons; i++){
			tempArray[i] = hiddenNeurons[i].gettopDownresponse();
		}
		//sort the top-down preResponses
		Arrays.sort(tempArray);
		cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
		if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
		cutOffValue = tempArray[cutOffPos];
		for (int i = 0; i < numNeurons; i++){
			if(hiddenNeurons[i].gettopDownresponse() < cutOffValue){
				hiddenNeurons[i].settopDownresponse(0);
			}
		}


		// Prescreen lateralExcitationResponse
		tempArray = new float[numNeurons];
		for (int i = 0; i < numNeurons; i++){
			tempArray[i] = hiddenNeurons[i].getlateralExcitationresponse();
		}
		//sort the lateral preResponses
		Arrays.sort(tempArray);
		cutOffPos = (int) Math.ceil((double)tempArray.length * (double)prescreenPercent);
		if(cutOffPos >= numNeurons-1) cutOffPos = numNeurons-1;
		cutOffValue = tempArray[cutOffPos];
		for (int i = 0; i < numNeurons; i++){
			if(hiddenNeurons[i].getlateralExcitationresponse() < cutOffValue){
				hiddenNeurons[i].setlateralExcitationresponse(0);
			}
		}

	}



	// Sort the topK elements to the beginning of the sort array where the index
	// of the top
	// elements are still in the pair.
	private static void topKSort(Pair[] sortArray, int topK) {

		for (int i = 0; i < topK; i++) {
			Pair maxPair = sortArray[i];
			int maxIndex = i;

			for (int j = i + 1; j < sortArray.length; j++) {
				// select temporary max value
				if (sortArray[j].value > maxPair.value) {

					maxPair = sortArray[j];
					maxIndex = j;

				}
			}
			// store the value of pivot (top i) element
			if (maxPair.index != i) {
				Pair temp = sortArray[i];
				// replace with the maxPair object
				sortArray[i] = maxPair;
				//replace maxPair index elements with the pivot
				sortArray[maxIndex] = temp;
			}
		}
	}

//	private void topKCompetition(int attentionId, boolean learn_flag, int type, float perfectmatch, DN2.MODE mode) {
//
//		// initializing the indexes
//		// winnerIndex is only for the winner neurons among the active neurons.
//		int winnerIndex = 0;
//
//		float[] copyArray = new float[usedNeurons];
//
//		// Pair is an object that contains the (index,response_value) of each
//		// hidden neurons.
//		Pair[] sortArray = new Pair[usedNeurons];
//
////		System.out.println("get neuron "+i+" response: "+hiddenNeurons[i].getnewresponse());
//		for(int j=0; j<usedNeurons; j++){
//			        if(areaMask[type-1][j]!=0){
//			            sortArray[j] = new Pair(j, hiddenNeurons[j].getnewresponse());
//			            copyArray[j] = hiddenNeurons[j].getnewresponse();
//			            preResponse[j] = hiddenNeurons[j].getnewresponse();
//
//			            hiddenNeurons[j].setnewresponse(0.0f);
//			            hiddenNeurons[j].setwinnerflag(false);
//			            inhibitoryNeurons[j].setwinnerflag(true);
//			        }
//			        else{
//			        	sortArray[j] = new Pair(j, 0);
//			            copyArray[j] = 0;
//			        }
//	     }
//
//
//		// Sort the array of Pair objects by its response_value in
//		// non-increasing order.
//		// The index is in the Pair, goes with the response ranked.
//		topKSort(sortArray, topK);
//
////		System.out.println("High top1 value of hidden: " + sortArray[0].value);
//
//		// check if the top winner has almost perfect match.
////		System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);
//
//		if (learn_flag) {
//			if (sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE)*perfectmatch && usedNeurons < numNeurons && perfectmatch>0.1f) { // add
//				System.out.println("Top 1 response: "+sortArray[0].value+", it index: "+sortArray[0].index);
//
//					hiddenNeurons[usedNeurons].setwinnerflag(true);
//					inhibitoryNeurons[usedNeurons].setwinnerflag(false);
//					hiddenNeurons[usedNeurons].setnewresponse(1.0f);// set to perfect match.
//					hiddenNeurons[usedNeurons].setType(type);
//				    if(type==2||type==3||type==6||type==7){
//						for (int j = 0; j < numNeurons; j++) {
//							hiddenNeurons[usedNeurons].setLateralMask(1, j);
//				        }
//					}
//					inhibitoryNeurons[usedNeurons].setType(type);
//					initializeRfMask(usedNeurons, attentionId, mode);
//					preResponse[usedNeurons] = 1-MACHINE_FLOAT_ZERO;
//				//	setwinnerIndex(0,hiddenNeurons[usedHiddenNeurons].getindex());
//					float[] temp1 = hiddenNeurons[sortArray[0].get_index()].getlocation();
//					float[] temp2 = new float [3];
//					temp2 = setnormvector(temp2);
//					float[] temp = {temp1[0]+temp2[0]*10*MACHINE_FLOAT_ZERO, temp1[1]+temp2[1]*10*MACHINE_FLOAT_ZERO, temp1[2]+temp2[2]*10*MACHINE_FLOAT_ZERO};
//					hiddenNeurons[usedNeurons].setlocation(temp);
//					areaMask[type-1][usedNeurons]=1.0f;
//					typeindex[type-1] += 1;
//					inhibitoryNeurons[usedNeurons].setLateralWeights(areaMask[type-1]);
//
//					for(int k=0; k<usedNeurons; k++){
//						if(inhibitoryNeurons[k].getType() == type){
//							inhibitoryNeurons[k].setLateralWeight(1,usedNeurons);
//						}
//					}
//					usedNeurons++;
//					winnerIndex++;
//			}
//		}
//
//		// identify the ranks of topk winners
//		float value_top1 = sortArray[0].value;
//		float value_topkplus1 = sortArray[topK].value; // this neurons is the
//														// largerst response
//														// neurons to be set to
//														// zero.
//
//		// Find the topK winners and their indexes.
//		while (winnerIndex < topK && perfectmatch>0.1f) {
//			// get the index of the top element.
//			int topIndex = sortArray[winnerIndex].get_index();
//		//	setwinnerIndex(winnerIndex,topIndex);
//			float tempnew= (copyArray[topIndex] - value_topkplus1)
//					/ (value_top1 - value_topkplus1 + MACHINE_FLOAT_ZERO);
//			hiddenNeurons[topIndex].setnewresponse(tempnew);
//			winnerIndex++;
//
//			hiddenNeurons[topIndex].setwinnerflag(true);
//			inhibitoryNeurons[topIndex].setwinnerflag(false);
//			System.out.println("winner neuron: "+ topIndex);
//		}
//		if(!learn_flag){
//		    System.out.println("********************************************");
//		}
//        System.out.println("used number of type "+type+" neurons: "+typeindex[type-1]);
//	}

	// TODO: AK: Check this function and see where it's used
	private void dynamic3x3topKCompetition(int attentionId, boolean learn_flag, int type,
										   DN2.MODE mode) {

		// Perform topk competition for each of the nine columns
		for (int maskId = 0; maskId < numBoxes; maskId++) {

			// do the topKcompetition
			float temp3 = (float)usedNeurons[maskId]/numNeuronPerColumn;
			int index = 0;
			if (mode == DN2.MODE.GROUP || mode == DN2.MODE.Speech || mode == DN2.MODE.NONE) {
				index = (int)(temp3/0.02f);
			}
			if(mode == DN2.MODE.MAZE){
				index = (int)(temp3/0.05f);
			}
			if(usedNeurons[maskId] == numNeuronPerColumn){
				index = index-1;
			}

			float percent =	mGrowthRate[index][type];
			topK = (int) Math.round(mGrowthRate[index][8] * numNeuronPerColumn);
			if(topK < 1) {
				topK = 1;
			}

			if(DN2.DEBUG) {
				System.out.println("TOPK : " + topK);
			}

			int currentIndex = 0;
			//float maxNum = mNeuronGrowth[currentIndex][0];
			float age = mNeuronGrowth[currentIndex][1];
			while(mAge >= age) {
				currentIndex++;
				if(currentIndex < mNeuronGrowth.length)
					age = mNeuronGrowth[currentIndex][1];
				else {
					currentIndex = mNeuronGrowth.length - 1;
					break;
				}
			}
			float maxNum = mNeuronGrowth[currentIndex][0];

			int bias = numNeuronPerColumn * maskId;
			// initializing the indexes
			// winnerIndex is only for the winner neurons among the active neurons.
			int numwinner = topK;
			boolean concept = true;
			float[] copyArray = new float[numNeuronPerColumn];

			// Pair is an object that contains the (index,response_value) of each
			// hidden neurons.
			Pair[] sortArray = new Pair[usedNeurons[maskId]];

//		System.out.println("usedNeurons: " + usedNeurons);

			for (int j = 0; j < usedNeurons[maskId]; j++) {
				if (areaMask[type - 1][bias+j] != 0) {
					// System.out.print("neuron"+j+" type "+type+" reached:
					// "+hiddenNeurons[j].getnewresponse()+" ");
					sortArray[j] = new Pair(j, hiddenNeurons[bias + j].getnewresponse());
					copyArray[j] = hiddenNeurons[bias + j].getnewresponse();
					preResponse[bias + j] = hiddenNeurons[bias + j].getnewresponse();
					hiddenNeurons[bias + j].setnewresponse(0.0f);
					hiddenNeurons[bias + j].setwinnerflag(false);
					//inhibitoryNeurons[bias + j].setwinnerflag(true);
				} else {
					sortArray[j] = new Pair(j, 0);
					copyArray[j] = 0;
				}
			}
			for (int i = usedNeurons[maskId]; i < numNeuronPerColumn; i++) {
				copyArray[i] = 0;
			}
			// Sort the array of Pair objects by its response_value in
			// non-increasing order.
			// The index is in the Pair, goes with the response ranked.
			topKSort(sortArray, topK);
						
//			System.out.println("High top1 value of hidden: " + sortArray[0].value);

			// check if the top winner has almost perfect match.
//			System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);
			if (type == 3) {
				concept = type3;
			} else {
				concept = true;
			}
			if (type == 4) {
				learn_flag = type4learn;
			}

			if (learn_flag) {
				// Adds a neuron
				if (sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE) * percent && usedNeurons[maskId] < numNeuronPerColumn
						&& percent > 0.001f && concept && temp3 <= maxNum) { // add
					if(DN2.DEBUG) {
						System.out.println("Top 1 response: " + sortArray[0].value + ", it index: " + sortArray[0].index);
					}

					// Make next new neuron a winner
					hiddenNeurons[bias + usedNeurons[maskId]].setwinnerflag(true);
					//inhibitoryNeurons[bias + usedNeurons[maskId]].setwinnerflag(false);
					hiddenNeurons[bias + usedNeurons[maskId]].setnewresponse(1.0f);// set to perfect match.
					hiddenNeurons[bias + usedNeurons[maskId]].setType(type); // Sets the type for the neuron
//				if (type == 2 || type == 3 || type == 6 || type == 7) {
//					for (int j = 0; j < numNeuronPerColumn; j++) {
//						hiddenNeurons[bias + usedNeurons[maskId]].setLateralMask(1, j);
//					}
//				}
					//inhibitoryNeurons[bias + usedNeurons[maskId]].setType(type);
					// Sets receptive field to be within the mask the neuron is included with
					//hiddenNeurons[bias+usedNeurons[maskId]].setBottomUpMasks(hiddenNeurons[bias+usedNeurons[maskId] - 1].getBottomUpMask());
					//initializeRfMask(bias + usedNeurons[maskId], attentionId, mode);
					preResponse[bias + usedNeurons[maskId]] = 1 - MACHINE_FLOAT_ZERO;
					// setwinnerIndex(0,hiddenNeurons[usedHiddenNeurons].getindex());
					float[] temp1 = hiddenNeurons[sortArray[0].get_index()].getlocation();
					float[] temp2 = new float[3];
					temp2 = setnormvector(temp2);
					float[] temp = { temp1[0] + temp2[0] * 10 * MACHINE_FLOAT_ZERO,
							temp1[1] + temp2[1] * 10 * MACHINE_FLOAT_ZERO,
							temp1[2] + temp2[2] * 10 * MACHINE_FLOAT_ZERO };
					hiddenNeurons[bias + usedNeurons[maskId]].setlocation(temp);

					// TODO: Fix setting this mask for the new neuron so that the neuron is not included in topk competition part below
					areaMask[type - 1][bias + usedNeurons[maskId]] = 1.0f;
					typeindex[type - 1] += 1;


					float[] tmp = new float[numNeuronPerColumn];

					System.arraycopy(areaMask[type - 1], bias, tmp, 0, numNeuronPerColumn);

//				inhibitoryNeurons[bias + usedNeurons[maskId]].setLateralWeights(tmp);
//				inhibitoryNeurons[bias + usedNeurons[maskId]].setLateralMasks(tmp);
//
//				for (int k = 0; k < usedNeurons[maskId]; k++) {
//					if (inhibitoryNeurons[bias + k].getType() == type) {
//						inhibitoryNeurons[bias + k].setLateralWeight(1, usedNeurons[maskId]);
//						inhibitoryNeurons[bias + k].setLateralMask(1, usedNeurons[maskId]);
//					}
//				}
					usedNeurons[maskId]++;
					numwinner--;
				}
			}


			if(numwinner!=0 && percent>0.001f && concept){

				int CountedWinners = 0;
				// TODO: AK: Make sure only k neurons fire (because need one loser)
				// TODO: AK: Make sure newly added (and non-updated) neuron isn't in this topk competition (The new one has BU weights all zeros anyway)
				for(int j=0; j< usedNeurons[maskId]; j++){
					if(areaMask[type-1][j] != 0){ //   && !(j == usedNeurons[type-1] && numwinner == topK-1) Doesn't allow newly added neuron to compete (its automatically top winner)
						// Neuron responses (in copyArray) decrease by the inhibitory neuron responses
						float[] tempResponse = new float[copyArray.length];

						System.arraycopy(copyArray, 0, tempResponse, 0, copyArray.length);//elementtWiseProduct(copyArray, inhibitoryNeurons[bias +j].getLateralMask());

						float response = tempResponse[j]; // AK replaced all copyArray[j] with this

						// TODO: Get rid of this. Arden added for debugging
						if(maskId == 0 && j==0) {
							int r = 1;
							int l = r;
						}

						// TODO: Begin new topk sorting logic

						// To be sorted list of the indices of the topk+1 neurons
						int[] IndicesTopK = new int[numwinner + 1];
						// To be sorted list of the responses of the topk+1 neurons
						float[] ResponsesTopK = new float[numwinner + 1];

						// Partial sort of topk + 1 neuron indices and responses
						for (int i=0; i < numwinner + 1; i++) {
							float max = tempResponse[0];
							int maxIndex = 0;
							for (int k=0; k < tempResponse.length - (i+1); k++) {
								if (tempResponse[k+1] > max) {
									max = tempResponse[k+1];
									maxIndex = k + 1;
								}
							}
							IndicesTopK[i] = maxIndex;
							ResponsesTopK[i] = tempResponse[maxIndex];
							tempResponse[maxIndex] = -1f;
						}

						float kIndex = 0;
						boolean isIn = false;
						for (int p = 0; p < numwinner; p++) {
							if (j == IndicesTopK[p]) {
								isIn = true;
								kIndex = p;
							}
						}

						// Let topk winners fire normalized responses
						if (isIn) { //response > ResponsesTopK[numwinner] || j == IndicesTopK[numwinner - 1]) {
							float tempnew = response;
							switch (responseCalculationType) {

								// Calculate response based on relative goodness of match values
								case GOODNESS_OF_MATCH:
									float loserResponse = ResponsesTopK[ResponsesTopK.length - 1];

									// If new neuron was added, use perfect match response (1.0) as top winner response value
									float r1 = ResponsesTopK[0];
									if (numwinner == topK - 1)
										r1 = 1.0f;

									// Scale response
									tempnew = (response - loserResponse) / (r1 - loserResponse + MACHINE_FLOAT_ZERO);
//									System.out.println("Response: " + response);
//									System.out.println("loserResponse: " + loserResponse);
//									System.out.println("r1: " + r1);
//									System.out.println("tempNew: " + tempnew);

									break;

								// Calculate response based on winner position in topk ranking
								case TRIANGLE_RESPONSE:
									// TODO: Check this!!
									// Ranks shift if new neuron added
									if (numwinner == topK - 1)
										kIndex++;

									// Scale response
									tempnew = 1.0f - (kIndex / topK);

									// TODO: Get rid of this. Arden used it for debugging
									if (maskId == 0 && DN2.DEBUG) {
										System.out.println("MASK: " + maskId);
										System.out.println("INDEX: " + j);
										System.out.println("RESPONSE: " + tempnew);
										int e = numUpdates;
										int t = e;
									}
									break;
							}

							hiddenNeurons[bias + j].setnewresponse(tempnew);

							hiddenNeurons[bias + j].setwinnerflag(true);
							//inhibitoryNeurons[bias + j].setwinnerflag(false);
						}
						// TODO: This ends the new topk sorting logic

						// Normalizes the topk responses
						// TODO: This should be partial sort. Sort only topk+1 numbers
				        /*Arrays.sort(tempResponse);
				        if(response > tempResponse[numNeuronPerColumn-numwinner-1]){
				        	CountedWinners++;

				        	float r1 = tempResponse[numNeuronPerColumn-1];
				        	if (numwinner == topK - 1)
				        		r1 = 1.0f;

				        	// TODO: AK: Make sure left part of denominator is 1.0 if new neuron added
				    	    float tempnew= (response - tempResponse[numNeuronPerColumn-numwinner-1])
									/ (r1 - tempResponse[numNeuronPerColumn-numwinner-1] + MACHINE_FLOAT_ZERO);
							hiddenNeurons[bias +j].setnewresponse(tempnew);

							// TODO: Get rid of this block, AK put it here for debugging
							if (maskId == 0) {
								System.out.println("MASK: " + maskId);
								System.out.println("INDEX: " + j);
								System.out.println("RESPONSE: " + tempnew);
								int r = numUpdates;
								int t = r;
							}

							hiddenNeurons[bias +j].setwinnerflag(true);
//							if(!learn_flag){
//								System.out.println("winner: neuron "+j+" the top: "+tempResponse[numNeuronPerColumn-numwinner]);
//							}
				            inhibitoryNeurons[bias +j].setwinnerflag(false);
				       }*/
					}
				}
			}
			int tmp=0;
			tmp++;
		}
		numUpdates++;
	}
//	private void dynamictopKCompetition(int attentionId, boolean learn_flag, int type, float percent, DN2.MODE mode) {
//
//		// initializing the indexes
//		// winnerIndex is only for the winner neurons among the active neurons.
//		int numwinner = topK;
//        boolean concept = true;
//		float[] copyArray = new float[numNeurons];
//
//		// Pair is an object that contains the (index,response_value) of each
//		// hidden neurons.
//		Pair[] sortArray = new Pair[usedNeurons];
//
//		for(int j=0; j<usedNeurons; j++){
//			        if(areaMask[type-1][j]!=0){
//	//		        	System.out.print("neuron"+j+" type "+type+" reached: "+hiddenNeurons[j].getnewresponse()+" ");
//			            sortArray[j] = new Pair(j, hiddenNeurons[j].getnewresponse());
//			            copyArray[j] = hiddenNeurons[j].getnewresponse();
//			            preResponse[j] = hiddenNeurons[j].getnewresponse();
//
//			            hiddenNeurons[j].setnewresponse(0.0f);
//			            hiddenNeurons[j].setwinnerflag(false);
//			            inhibitoryNeurons[j].setwinnerflag(true);
//			        }
//			        else{
//			            sortArray[j] = new Pair(j, 0);
//			            copyArray[j] = 0;
//			        }
//	     }
//		for(int i=usedNeurons; i<numNeurons;i++){
//			 copyArray[i] = 0;
//		}
//		// Sort the array of Pair objects by its response_value in
//		// non-increasing order.
//		// The index is in the Pair, goes with the response ranked.
//		topKSort(sortArray, topK);
//
////		System.out.println("High top1 value of hidden: " + sortArray[0].value);
//
//		// check if the top winner has almost perfect match.
////		System.out.println(sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE+1.0f*lateralPercent) && usedHiddenNeurons < numNeurons);
//		 if(type == 3){
//				concept = type3;
//		 }
//		 else{
//				concept = true;
//		 }
//		 if(type == 4){
//				learn_flag = type4learn;
//		 }
//
//		if (learn_flag) {
//			if (sortArray[0].value < (ALMOST_PERFECT_MATCH_RESPONSE)*percent && usedNeurons < numNeurons && percent>0.001f && concept) { // add
//				   System.out.println("Top 1 response: "+sortArray[0].value+", it index: "+sortArray[0].index);
//
//				    hiddenNeurons[usedNeurons].setwinnerflag(true);
//				    inhibitoryNeurons[usedNeurons].setwinnerflag(false);
//				    hiddenNeurons[usedNeurons].setnewresponse(1.0f);// set to perfect match.
//				    hiddenNeurons[usedNeurons].setType(type);
//				    if(type==2||type==3||type==6||type==7){
//						for (int j = 0; j < numNeurons; j++) {
//							hiddenNeurons[usedNeurons].setLateralMask(1, j);
//						}
//				    }
//				    inhibitoryNeurons[usedNeurons].setType(type);
//				    initializeRfMask(usedNeurons, attentionId, mode);
//				    preResponse[usedNeurons] = 1-MACHINE_FLOAT_ZERO;
//				//	setwinnerIndex(0,hiddenNeurons[usedHiddenNeurons].getindex());
//					float[] temp1 = hiddenNeurons[sortArray[0].get_index()].getlocation();
//					float[] temp2 = new float [3];
//					temp2 = setnormvector(temp2);
//					float[] temp = {temp1[0]+temp2[0]*10*MACHINE_FLOAT_ZERO, temp1[1]+temp2[1]*10*MACHINE_FLOAT_ZERO, temp1[2]+temp2[2]*10*MACHINE_FLOAT_ZERO};
//					hiddenNeurons[usedNeurons].setlocation(temp);
//
//					areaMask[type-1][usedNeurons]=1.0f;
//					typeindex[type-1] += 1;
//					inhibitoryNeurons[usedNeurons].setLateralWeights(areaMask[type-1]);
//					inhibitoryNeurons[usedNeurons].setLateralMasks(areaMask[type-1]);
//
//					for(int k=0; k<usedNeurons; k++){
//						if(inhibitoryNeurons[k].getType() == type){
//							inhibitoryNeurons[k].setLateralWeight(1,usedNeurons);
//							inhibitoryNeurons[k].setLateralMask(1,usedNeurons);
//						}
//					}
//					usedNeurons++;
//					numwinner--;
//
//			}
//		}
//
//
//
//		if(numwinner!=0 && percent>0.001f && concept){
//
//            for(int j=0; j< usedNeurons; j++){
//                if(areaMask[type-1][j] != 0){
//				        float[] tempResponse = elementWiseProduct(copyArray, inhibitoryNeurons[j].getLateralMask());
////				        System.out.println("mask: "+inhibitoryNeurons[j].getLateralMask()[0]+" "+inhibitoryNeurons[j].getLateralMask()[1]+" "+inhibitoryNeurons[j].getLateralMask()[2]);
////				        System.out.println("product: "+tempResponse[0]+" "+tempResponse[1]+" "+tempResponse[2]);
////				        System.out.println("res: "+copyArray[0]+" "+copyArray[1]+" "+copyArray[2]);
//				        Arrays.sort(tempResponse);
//				        if(copyArray[j] >= tempResponse[numNeurons-numwinner]){
//				    	    float tempnew= (copyArray[j] - tempResponse[numNeurons-numwinner-1])
//									/ (tempResponse[numNeurons-1] - tempResponse[numNeurons-numwinner-1] + MACHINE_FLOAT_ZERO);
//							hiddenNeurons[j].setnewresponse(tempnew);
//
//							hiddenNeurons[j].setwinnerflag(true);
////							if(!learn_flag){
//								System.out.println("winner: neuron "+j+" the top: "+tempResponse[numNeurons-numwinner]);
////							}
//				            inhibitoryNeurons[j].setwinnerflag(false);
//				       }
//				}
//            }
//	    }
//        System.out.println("Used number of type "+type+" neurons: "+typeindex[type-1]);
//
//	}

	//convert the new responses to old responses
	public void replaceHiddenLayerResponse() {
		for(int j=0;j<numNeurons; j++){
			hiddenNeurons[j].replaceResponse();
		}
	}

	//set y attend information
	public void setYattend(boolean y){
		yAttend = y;
	}

	//get the number of winner y neurons
	public int getTopK() {
		return topK;
	}

	//set the number of winner y neurons
	public void setTopK(int topK) {
		this.topK = topK;
	}

	public int getNumBottomUpWeights() {
		return numBottomUpWeights;
	}

	public void setNumBottomUpWeights(int numBottomUpWeights) {
		this.numBottomUpWeights = numBottomUpWeights;
	}

	public int getNumTopDownWeights() {
		return numTopDownWeights;
	}

	public void setNumTopDownWeights(int numTopDownWeights) {
		this.numTopDownWeights = numTopDownWeights;
	}

	public int getNumNeurons() {
		return numNeurons;
	}

	public int gettypeNum(){
		return numTypes;
	}

	public void set5growthrate(boolean a){
		type5g = a;
	}
	public void setConcept(boolean c3){
		type3 = c3;
	}

	public void set5Learning(boolean l5){
		type5learn = l5;
	}

	public void set4Learning(boolean l4){
		type4learn = l4;
	}

	public void setPriLateralvector(float[] vector) {
		for(int i = 0; i < vector.length; i++) {
			priLateralvector[i] = vector[i];
		}
	}

//	public void setBottomupMask(float[] mask, int index){
//		hiddenNeurons[index].setBottomUpMasks(mask);
//	}

//	public void setTopdownMask(float[] mask, int index){
//		hiddenNeurons[index].setTopDownMasks(mask);
//	}

//    //get the local receptive field
//	public float[][] getRfMask() {
//		float[][] bottomUpMask = new float [numNeurons][numBottomUpWeights];
//		for(int i = 0; i < numNeurons; i++){
//			System.arraycopy(hiddenNeurons[i].getBottomUpMask(),0,bottomUpMask[i],0,numBottomUpWeights);
//		}
//		return bottomUpMask;
//	}

	//get the number of y neurons
	public int getUsedHiddenNeurons(int maskId) {
		return usedNeurons[maskId];
	}
	//re-set responses for each neuron
	public void resetResponses() {

		for(int i = 0; i < numNeurons; i++){
			hiddenNeurons[i].resetResponses();
		}

	}

	//multiply elements between 2 vectors
	public float[] elementWiseProduct(float[] vec1, float[] vec2) {
		assert vec1.length == vec2.length;
		int size = vec1.length;
		float[] result = new float[size];
		for (int i = 0; i < size; i++) {
			result[i] = vec1[i] * vec2[i];
		}
		return result;
	}

	public class Pair implements Comparable<Pair> {
		public final int index;
		public final float value;

		public Pair(int index, float value) {
			this.index = index;
			this.value = value;
		}

		public int compareTo(Pair other) {
			return -1 * Float.valueOf(this.value).compareTo(other.value);
		}

		public int get_index() {
			return index;
		}
	}
	//calculate the mean value for the non-zero elements in a vector
	public static float mean(float[] m, float[] mask) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (mask[i] != 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}

	//calculate the mean value for a vector
	public static float mean(float[] m) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (m[i] > 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}

	public float[] setnormvector(float[] vector){
		Random ran = new Random();
		for(int i = 0; i < vector.length; i++){
			vector[i] = ran.nextFloat();
		}
		norm(vector);
		return vector;
	}

	public void setNumNeurons(int numNeurons) {
		this.numNeurons = numNeurons;
	}
	/*
	public int[] getwinnerIndexs() {
		return winnerIndexs;
	}

	public void setwinnerIndex(int topk, int index) {
		this.winnerIndexs[topk] = index;
	}*/

	//calculate the mean value
	public float mean(float[] m, int[] length) {
		float sum = 0;
		float count = 0;
		int temp = 0;
		for (int i = 0; i < length.length; i++) {
			if(length[i]!=0){
				for(int j=temp*numNeurons; j<temp*numNeurons+length[i]; j++){
					if (m[j] > 0) {
						sum += m[j];
						count++;}
				}
				temp++;
			}
		}
		return sum / count;
	}

	//L-2 normalization
	public float[] norm(float[] weight){
		float norm = 0;
		int size = weight.length;
		for (int i = 0; i < size; i++){
			norm += weight[i]* weight[i];
		}
		norm = (float) Math.sqrt(norm);
		if (norm > 0){
			for (int i = 0; i < size; i++){
				weight[i] = weight[i]/norm;
			}
		}
		return weight;
	}
	//find the top-k neighbors for each glial cell
	public void topKNeighborSort(Pair[] sortArray, int topK){

		for (int i = 0; i < topK; i++) {
			Pair minPair = sortArray[i];
			int minIndex = i;

			for (int j = i+1; j < sortArray.length; j++) {

				if(sortArray[j].value < minPair.value){ // select temporary max
					minPair = sortArray[j];
					minIndex = j;

				}
			}

			if(minPair.index != i){
				Pair temp = sortArray[i]; // store the value of pivot (top i) element
				sortArray[i] = minPair; // replace with the maxPair object.
				sortArray[minIndex] = temp; // replace maxPair index elements with the pivot.
			}
		}
	}

	//pull the y neurons locations away
	public void pullneurons(float pullrate){
		for(int cl=0;cl<numBoxes;cl++) {

			int minvalue;
			if(usedNeurons[cl] < glialcells[0].gettopk()){
				minvalue = usedNeurons[cl];
			}
			else{
				minvalue = glialcells[0].gettopk();
			}
			for(int i = 0;i < numGlial; i++){
				Pair[] distances =new Pair [usedNeurons[cl]];

				for(int k=0; k<usedNeurons[cl]; k++){
					distances[k] = new Pair(k,glialcells[i].computeDistance(hiddenNeurons[numNeuronPerColumn*cl+k].getlocation()));
				}


				topKNeighborSort(distances,minvalue);
				for(int k = 0;k < minvalue; k++){
					glialcells[i].setpullindex(k, distances[k].index);
					float[] dis =new float[3];
					for(int m = 0; m < 3; m++){
						dis[m]=glialcells[i].getlocation()[m] - hiddenNeurons[numNeuronPerColumn*cl+distances[k].index].getlocation()[m];
					}
					glialcells[i].setpullvector(dis,k);
				}
			}
			for(int k =0; k<usedNeurons[cl]; k++){
				float[] pullvector = new float [3];
				for(int l = 0;l < numGlial; l++){
					for(int h = 0;h < minvalue; h++){
						if(glialcells[l].getpullindex(h) == hiddenNeurons[numNeuronPerColumn*cl+k].getindex()){
							for(int m = 0; m < 3; m++){
								pullvector[m] += glialcells[l].getpullvector(h)[m];
							}
						}
					}
				}
				float[] temp = new float[3];
				pullvector = norm(pullvector);
				for(int m = 0; m < 3; m++){
					temp[m] = hiddenNeurons[numNeuronPerColumn*cl+k].getlocation()[m]+pullrate*pullvector[m];
				}
				hiddenNeurons[k].setlocation(temp);
			}
		}
	}

	//calculate the distances between other neurons and record the indexs of neighbors
	public void calcuateNeiDistance() {
		for(int cl=0;cl<numBoxes;cl++) {

			Pair[][] dist = new Pair[usedNeurons[cl]][usedNeurons[cl]-1];
			for(int i =0; i < usedNeurons[cl]; i++){
				for(int j =i+1; j < usedNeurons[cl]; j++){
					float dis = hiddenNeurons[numNeuronPerColumn*cl+i].computeDistance(hiddenNeurons[numNeuronPerColumn*cl+j].getlocation());
					dist[i][j-1] = new Pair(j,dis);
					dist[j][i] = new Pair(i,dis);
				}
				topKNeighborSort(dist[i],addNum);
				int[] NeiIndex = new int[addNum];
				for(int k = 0; k < addNum; k++) {
					NeiIndex[k] = dist[i][k].get_index();
				}
				hiddenNeurons[numNeuronPerColumn*cl+i].setneiIndex(NeiIndex);
			}
		}
	}

	//add new lateral connections for each neuron
	public void growLateralConnection(int j, int[] list) {
		for(int i = 0; i < list.length; i++) {
			for(int k = 0; k < addNum; k++) {
				int index = hiddenNeurons[i].getneiIndex()[k];
				boolean g = false;
				int t = 0;
				//add the neuron's neighbor lateral connection's neighbor (j-1)-th in the connections
				if ((hiddenNeurons[j].InlateralMask(index) == false)&&(hiddenNeurons[j].InlateralDeletelist(index) == false)) {
					g = true;
					hiddenNeurons[j].addLateralMasks(index);

				}

/*				while(g) {
					if(t <1) {
						System.out.println("excuted!");
						System.out.println("neuron "+j+" add neuron "+i+"'s neighbor: neuron "+ index);
						t += 1;
						}
					if(getKey()) {
						g = false;
					}
				}*/
			}
		}
	}

	//send the y neurons' weights
	public float[][] send_y(int display_num, int display_start_id, int type){
		//get the specific y neurons
		float[][] weights;
		int start_id = display_start_id - 1;
		if (start_id < 0)
			start_id = 0;
		if (start_id >= numNeurons)
			start_id = 0;
		int end_id = start_id + display_num;
		if (end_id > numNeurons)
			end_id = numNeurons;
		if (end_id < 0)
			end_id = numNeurons;

		// bottom up weight
		if (type == 1) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getBottomUpWeights()[j];
				}
			}
			return weights;
		}

		// bottom up age
		else if (type == 2) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].getbottomupage()[j];
				}
			}
			return weights;
		}

		// bottom up mask
		else if (type == 3) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].getBottomUpMask()[j];
				}
			}
			return weights;
		}

		// bottom up variance
		else if (type == 4) {
			weights = new float[display_num][numBottomUpWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numBottomUpWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].getBottomUpVariances()[j];
				}
			}
			return weights;
		}

		// topDown weight
		else if (type == 5) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					weights[i][j] = hiddenNeurons[i].getTopDownWeights()[j];
				}
			}
			return weights;
		}

		// topDown age
		else if (type == 6) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].gettopdownage()[j];
				}
			}
			return weights;
		}

		// topDown mask
		else if (type == 7) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].getTopDownMask()[j];
				}
			}
			return weights;
		}

		// topDown variance
		else if (type == 8) {
			weights = new float[display_num][numTopDownWeights];
			for (int i = start_id; i < end_id; i++) {
				for (int j = 0; j < numTopDownWeights; j++) {
					//weights[i][j] = hiddenNeurons[i].getTopDownVariances()[j];
				}
			}
			return weights;
		}
		//lateral excitation weights
		if (type == 9) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
				for(int j = 0; j < numNeurons; j++){
					weights[i][j] = hiddenNeurons[i].getLateralWeights()[j];
				}
			}
			return weights;
		}
//		//lateral excitation ages
//		if (type == 10) {
//			weights = new float[display_num][numNeurons];
//			for (int i = start_id; i < end_id; i++) {
//			    for(int j = 0; j < numNeurons; j++){
//			    	weights[i][j] = hiddenNeurons[i].getlateralage()[j];
//			    }
//		    }
//			return weights;
//		}
		//lateral excitation masks
		if (type == 11) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
				for(int j = 0; j < numNeurons; j++){
					weights[i][j] = hiddenNeurons[i].getLateralMask()[j];
				}
			}
			return weights;
		}
//		//lateral excitation variances
//		if (type == 12) {
//			weights = new float[display_num][numNeurons];
//			for (int i = start_id; i < end_id; i++) {
//			    for(int j = 0; j < numNeurons; j++){
//			    	weights[i][j] = hiddenNeurons[i].getLateralVariances()[j];
//			    }
//		    }
//			return weights;
//		}
		//inhibition weights
		if (type == 13) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
				for(int j = 0; j < numNeurons; j++){
					//weights[i][j] = inhibitoryNeurons[i].getLateralWeights()[j];
				}
			}
			return weights;
		}
		//inhibition areas
		if (type == 14) {
			weights = new float[display_num][numNeurons];
			for (int i = start_id; i < end_id; i++) {
				for(int j = 0; j < numNeurons; j++){
					//weights[i][j] = inhibitoryNeurons[i].getLateralMask()[j];
				}
			}
			return weights;
		}
		return null;
	}

	public int sumConnections() {
		int result = 0;

		for(int cl=0;cl<numBoxes;cl++) {

			for(int i=0; i<usedNeurons[cl]; i++) {
				//result += sumArray(hiddenNeurons[i].getBottomUpMask());
				result += sumArray(hiddenNeurons[i].getLateralMask());
				//result += sumArray(hiddenNeurons[i].getTopDownMask());
			}
		}
		return result;
	}

	public int sumArray(float[] a) {
		int result = 0;
		for(int i=0; i<a.length; i++) {
			if(a[i] != 0) {
				result += 1;
			}
		}
		return result;
	}

	protected void Setup(int initialNumNeurons, int sensorSize, int motorSize, int[][] rfSizes, int[][] rfStrides, int numBoxes,
						 int[][] rf_id_loc, int[][] inputSize, float prescreenPercent, int[] typeNum, float[][] growthTable, float[][] meanTable,
						 boolean dynamicinhibition, float[][] neuronGrowth, DNHandler.RESPONSE_CALCULATION_TYPE _responseCalculationType) {

		this.numBoxes = numBoxes;

		//set response type
		responseCalculationType = _responseCalculationType;
		//topK=1;
		// TODO: AK added this
		this.setTopK(topK);
		//winnerIndexs = new int [topK];
//		this.usedHiddenNeurons = topK + 1; // bound of used neurons.
		//get used neuron types
		typeindex = new int[typeNum.length];
		for(int i=0; i<typeNum.length; i++){
			typeindex[i] = typeNum[i];
		}
		for(int i = 0; i < typeindex.length; i++){
			numTypes += typeindex[i]/(topK + 1);
		}

		isPriLateral = true;
		priLateralvector = new float[PriLateralsize];
		this.addNum = 1;
		//initialize the number of used neurons
		this.usedNeurons = new int[numBoxes];
		for(int i=0;i<numBoxes;i++)
			this.usedNeurons[i] = (topK+1)*numTypes; // 2 * numTypes
		//TODO: AK: Figure out if this needs to be topk+1 * numTypes not 2*numTypes (Made this change above and below)
		//TODO: AK: Or if it should just check that there are enough neurons and be initialNumNeurons only
		//initialize the number of total neurons
		this.numNeuronPerColumn = initialNumNeurons; // AK Changed this so that numneurons specified is the exact number of neurons//  + (topK+1) * numTypes; //2*numTypes;
		this.numNeurons= numNeuronPerColumn*numBoxes;
		//whether use dynamic inhibition or not
		this.dynamicInhibition = dynamicinhibition;
		//size of local receptive field
		this.rfSize = rfSizes;
		//moving step of local receptive field
		this.rfStride = rfStrides;
		//array of generated local receptive fields
		this.rf_id_loc = rf_id_loc;
		//initialize the input size
		this.inputSize = inputSize[0];
		//get the grow rate table values
		this.mGrowthRate = new float[growthTable.length][];
		for(int i = 0; i < growthTable.length; i++){
			mGrowthRate[i] = new float[growthTable[i].length];
			System.arraycopy(growthTable[i], 0, mGrowthRate[i], 0, growthTable[i].length);
		}

		this.mNeuronGrowth = new float[neuronGrowth.length][];
		for(int i = 0; i < neuronGrowth.length; i++){
			mNeuronGrowth[i] = new float[neuronGrowth[i].length];
			System.arraycopy(neuronGrowth[i], 0, mNeuronGrowth[i], 0, neuronGrowth[i].length);
		}

		//get the mean value table values
		this.mMeanValue = new float[meanTable.length][];
		for(int i = 0; i < meanTable.length; i++){
			mMeanValue[i] = new float[meanTable[i].length];
			System.arraycopy(meanTable[i], 0, mMeanValue[i], 0, meanTable[i].length);
		}


		//construct the area mask array
		areaMask = new float[7][numNeurons];

		//initialize the neurons

		// TODO: AK: Get rid of the magic numbers in this code.
		//initialize each initial neuron
		for (int cl=0;cl<numBoxes;cl++) { // For each of the 9 columns in 3dEye
			{
				int topkPlusOne = topK + 1;
				int bias = cl*numNeuronPerColumn;

				//set the percentage of lateral excitation responses
				this.lateralPercent = 1.0f;

				this.prescreenPercent = prescreenPercent;
				//set the pre-response values
				this.preResponse = new float[numNeurons];
				for(int i=0; i<numNeuronPerColumn; i++){
					preResponse[bias+i] = 0;
				}

				numBottomUpWeights = sensorSize;
				numTopDownWeights = motorSize;

				// TODO: AK: Fix the area mask so it is outside this loop on the masks and sets up topk+1 neurons with ones
				//construct the area mask array
				//areaMask = new float[7][numNeurons];
				//initialize the area mask
				int tempindex = 0;
				for(int i=0; i<7; i++){ // For each neuron type
					for(int j=0; j<numNeuronPerColumn; j++){
						areaMask[i][bias+j] = 0;
						if (typeindex[i] !=0 && j < topkPlusOne) {
							areaMask[i][bias+j] = 1.0f;
						}
					}
					/*if(typeindex[i] !=0){
						areaMask[i][bias+2*tempindex]=1.0f;
						areaMask[i][bias+2*tempindex+1]=1.0f;
						tempindex += 1;
					}*/
				}
				yAttend = false;
				type3 = true;
				type4learn = true;
				type5learn = true;
				type5g =true;

				//initialize the glial cells
				//set the number of neighbor neurons to be pulled
				glialtopK = 3;
				//set the size of glial cells
				glialsize = 10;
				numGlial = glialsize*glialsize*glialsize;
				glialcells = new Glial[numGlial];
				for(int i=0; i<glialsize;i++){
					for(int j=0; j<glialsize;j++){
						for(int k=0; k<glialsize;k++){
							float[] temp = {(float)k,(float)j,(float)i};
							glialcells[glialsize*glialsize*i+glialsize*j+k] = new Glial(glialtopK,glialsize*glialsize*i+glialsize*j+k);
							glialcells[glialsize*glialsize*i+glialsize*j+k].setlocation(temp);
						}
					}
				}



			}
		}

	}

	//added by jacob for getting firing ages
	public float getFiringNeuronsAges(int i){
		return hiddenNeurons[i].getfiringage();
	}

	//added by jacob for getting firing neuron type
	public int getFiringNeuronType(int i){
		return hiddenNeurons[i].getType();
	}

	//added by jacob for top down weights
	public float[] getTopDownWeights(int i){
		return hiddenNeurons[i].getTopDownWeights();
	}

	//added by jacob for lateral  weights
	public float[] getLateralWeights(int i){
		return hiddenNeurons[i].getLateralWeights();
	}

	//added by jacob for bottom up weights
	public float[] getBottomUpWeights(int i){
		return hiddenNeurons[i].getBottomUpWeights();
	}

	// convert into 1d Array
	public float[] getNewResponse1D() {
		//construct the new array
		float[] inputArray = new float[hiddenNeurons.length];

		int offset = 0;
		for(int i = 0; i < usedNeurons.length; i++) {
			for (int j = 0; j < numNeuronPerColumn; j++) {
				inputArray[offset] = hiddenNeurons[offset].getnewresponse();
				offset++;
			}
		}

//		System.out.println(Arrays.toString(inputArray));

		return inputArray;
	}

	//added by jacob
	public boolean[] getWinnerFlags (){

		boolean[] winnerFlags = new boolean[hiddenNeurons.length];

		for (int i = 0; i < hiddenNeurons.length; i++){
			winnerFlags[i] = hiddenNeurons[i].getwinnerflag();
		}

		return winnerFlags;
	}

	/**
	 * Only called by RS_HiddenLayer
	 */
	protected HiddenLayer() {}

	public void loadAllocations()
	{

	}

//	public void reinitializeRenderScript(RenderScript rs)
//	{
//
//	}

	public void finish(){}
	public void saveAllocations() {}

	public void setSerotoninMultiplier(float multiplier){
		for (Neuron hiddenNeuron : hiddenNeurons) {
			hiddenNeuron.setSerotoninMultiplier(multiplier);
		}
	}

	public void setDopamineMultiplier(float multiplier){
		for (Neuron hiddenNeuron : hiddenNeurons) {
			hiddenNeuron.setDopamineMultiplier(multiplier);
		}
	}
}

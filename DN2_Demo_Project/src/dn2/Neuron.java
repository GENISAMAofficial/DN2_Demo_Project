package dn2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;


/**
 * <h1> Neuron </h1>
 *
 * <p>
 * 	This class contains all the information relevant to a neuron for each Layer in the Developmental Network (DN) algorithm.
 *  Each neuron will store its firing age, 3D space location, bottomup, topdown, or lateral response depending on which Layer the neuron is initialized.
 *  For example, Sensor (X) neuron will only have topdown weights because it can only connect to other Y neuron, prediction in X is not currently used in this implementation.
 *
 * <\p>
 *
 */
public class Neuron implements Serializable{
	// test comment
	/**
	 * Class serial ID.
	 *
	 * Identifies the array of bytes of the object sent across network sockets or text files.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Neuron's state
	 *
	 * <ul>
	 *  <li> 0 for initial </li>
	 * 	<li> 1 for learning </li>
	 * </ul>
	 */
	private boolean mState;

	/** Neuron's firing age */
	short mFiringAge;

	/** Neuron's type (only applicable for Y neurons) */
	protected int mType;

	/**
	 * This field determines whether the neuron is for the MotorLayer or HiddenLayer.
	 *
	 * <ul>
	 * 	<li> True = Y neuron </li>
	 *  <li> False = Z neuron </li>
	 * </ul>
	 */
	private boolean mCategory;

	/** Neuron's index */
	protected int mIndex;

	/**
	 * This field determines whether the neuron is a winner from the topk competition.
	 */
	protected boolean winnerFlag;

	/** Neuron's 3D location */
	private float[] mLocation;

	/** Bottom-up preResponse value */
	protected float bottomUpResponse;

	/** Top-down preResponse */
	protected float topDownResponse;

	/** Lateral preResponse value */
	private float lateralResponse;

	/** Computed total response at time t */
	protected float newResponse;

	/** Computed total response at time t-1 (previous time frame) */
	private float oldResponse;

	// Arden added for motivation
	private float serotoninLevel = 0;
	private float dopamineLevel = 0;
	private float mSweetInput = 0;
	private float mPainInput = 0;
	private float serotoninMultiplier = 0;
	private float dopamineMultiplier = 0;

	/** Bottom-up weights */
	private float[] mBottomUpWeights;

	/** Top-down weights */
	private float[] mTopDownWeights;

	//each neuron's lateral weights
	//private float[] mLateralWeights;

	/** Lateral weights. This array updates for every Y neuron created. */
	private ArrayList<Pair> mLateralWeight;

	/** Record each neuron's variances between bottom-up weights and bottom-up inputs */
	//private float[] mBottomUpVariance;


	/** Record each neuron's variances between top-down weights and top-down inputs */
	//private float[] mTopDownVariance;

	/** Record each neuron's variances between lateral weights and lateral inputs */
	//private float[] mLateralVariance;

	/** Bottom-up mask. This determines which bottom-up inputs are to be considered during topk competitions */
	//private float[] mBottomUpMask;

	/** Top-down mask. This determines which bottom-up inputs are to be considered during topk competitions */
	//private float[] mTopDownMask;

	//each neuron's lateral mask
	//private float[] mLateralMask;

	/** Lateral mask. This determines which bottom-up inputs are to be considered during topk competitions */
	private ArrayList<Integer> mLateralMasks;

	/** Bottom-up synapse's age of the neuron */
	//private short[] bottomUpAge;

	/** Top-down synapse's age of the neuron */
	//private short[] topDownAge;

	//each lateral synapse's age of the neuron
	//private int[] lateralAge;

	///** Lateral synapse's age of the neuron */
	//private ArrayList<Integer> lateralAges;



	/** The parameter for lateral preResponse's contribution */
	private float lateralRatio;

	/** Record the size of input */
	private int[] inputSize;

	/** The parameter of learning rate */
	private final float GAMMA = 2000;

	/** Constant used to determine when a floating-point value is considered as zero */
	private static float MACHINE_FLOAT_ZERO;

	/** The parameter of learning rate */
	private final int T1 = 20;

	/** The parameter of learning rate */
	private final int T2 = 200;

	/** */
	private final float C = 2;

	// Parameters for neurotransmitter learning rate
	private final int T1_mod = 0;
	private final int T2_mod = 1;
	private final float GAMMA_mod = 5;

	// SM -> Synapse Maintenance

	/**
	 * Synaptic Maintenance age value. If the neurons firing age is larger than this constant,
	 * then the neuron can be considered for synaptic maintenance (grow or cut).
	 */
	//TODO: Arden changed this value so synaptic maintanance is off
	protected  int SMAGE = 1000000000;

	/** The parameter for synaptic maintenance. */
	private final float SMUPPERTHRESH = 1.5f;

	/** The parameter for synaptic maintenance. */
	private final float SMLOWERTHRESH = 0.9f;

	/** Number of Lateral neurons */
	protected int mLateralSize;

	/** */
	private int[] neiIndex;

	/** */
	private boolean isGrowLateral;

	/** */
	private ArrayList<Integer> lateralGrowlist;

	/** */
	private ArrayList<Integer> lateralDeletelist;

	protected int mBottomUpSize;
	protected int mTopDownSize;

	// TODO: Arden added these for visualization
	public float mMean = 0;
	public float mNorm = 1;

	//float maxVolume = 0;
	public static float volumeContribution = 0.1f;

	public float[] mLateralWeights;

	private int numMotor;

	//construct neuron
	// This one knows the type and input size for the neuron

	/**
	 * This constructor will initialize a neuron instance with a known neuron type and input size.
	 *
	 * @param bottomupsize
	 * @param topdownsize
	 * @param lateralsize
	 * @param category
	 * @param type
	 * @param index
	 * @param inputsize
	 */
	public Neuron(int bottomupsize, int topdownsize, int lateralsize, boolean category, int type, int index, int[] inputsize, int numMotor) {

		this.numMotor = numMotor;
		//set neuron's index
		mIndex = index;
		//set neuron's type: 1 for y neuron; 0 for z neuron
		mType = type;
		//initialize neuron's category; 1 for y neuron; 0 for z neuron
		mCategory = category;
		//initialize neuron's firing age
		mFiringAge = 0;
		//initialize neuron's winner flag
		winnerFlag = false;
		//initialize response
		newResponse = 0.0f;
		//initialize response
		oldResponse = 0.0f;
		//set the
		mLateralSize = lateralsize;
		//set the machine zero
		MACHINE_FLOAT_ZERO = 0.00001f;
		//set the input size
		inputSize = new int[inputsize.length];
		for (int i = 0; i < inputsize.length; i++) {
			inputSize[i] = inputsize[i];

		}
		//construct the 3D location vector
		mLocation = new float[3];
		//construct each neuron's bottom-up weight vector
		mBottomUpWeights = new float[bottomupsize+1+numMotor*2]; // +8 elements for serotonin and dopamine
		//construct each neuron's top-down weight vector
		mTopDownWeights = new float[topdownsize];
		//initialize the z neuron's lateral mask
		mLateralWeight = new ArrayList<Pair> ();
		//initialize the variances between lateral inputs and lateral weights
		//mLateralVariance = new float[lateralsize];
		if (mCategory == false && lateralsize > 0) {
			//initialize the lateral mask
//			mLateralMask = new float[lateralsize];
			mLateralMasks = new ArrayList<Integer>();
			//lateralAges = new ArrayList<Integer>();
			for (int i = 0; i < lateralsize; i++) {
//				mLateralMask[i] = 1.0f;
				mLateralMasks.add(i);
				//lateralAges.add(0);
				Pair a = new Pair(i,0);
				mLateralWeight.add(a);
			}

			//initialize the lateral synapses' ages
//			lateralAge = new int[lateralsize];
		}
		//initialize weights for y neuron
		if (mCategory) {
			//set machine zero
			MACHINE_FLOAT_ZERO = 0.0001f;
			mLateralMasks = new ArrayList<Integer>();
			//lateralAges = new ArrayList<Integer>();
			lateralGrowlist = new ArrayList<Integer>();
			lateralDeletelist = new ArrayList<Integer>();
			//randomly initialize y neuron's bottom-up weights for the neurons which have bottom-up connections
			if (mType == 4 || mType == 5 || mType == 6 || mType == 7) {
				initializeWeight(mBottomUpWeights, bottomupsize);
			}
			//set the bottom-up weights to be 0 for the neurons which don't have bottom-up connections
			else {
				for (int i = 0; i < bottomupsize; i++) {
					mBottomUpWeights[i] = 0;
				}
			}
			//initialize y neuron's top-down weights for the neurons which have top-down connections
			if (mType == 1 || mType == 3 || mType == 5 || mType == 7) {
				initializeWeight(mTopDownWeights, topdownsize);
			}
			//set the top-down weights to be 0 for the neurons which don't have top-down connections
			else {
				for (int i = 0; i < topdownsize; i++) {
					mTopDownWeights[i] = 0;
				}
			}
			//initialize y neuron's lateral weights for the neurons which have lateral connections
			if (mType == 2 || mType == 3 || mType == 6 || mType == 7) {
				float[] lateralWeights = new float[mLateralSize];
				initializeWeight(lateralWeights, lateralsize);
				for(int i=0; i<mLateralSize; i++){
					mLateralMasks.add(i);
					//lateralAges.add(0);
					Pair a = new Pair(i,lateralWeights[i]);
					mLateralWeight.add(a);
				}
			}
			//set the lateral weights to be 0 for the neurons which don't have lateral connections
/*			else{
			    for (int i = 0; i < lateralsize; i++) {
				    mLateralWeights[i] = 0;
			    }
			}*/

			//construct bottom-up mask vector
			//mBottomUpMask = new float[bottomupsize+1];
			// TODO: Check if fix in hidden layer fixed mask initialization issue
			/*for (int i = 0; i < bottomupsize; i++) {
				mBottomUpMask[i] = 1f;
			}*/
			//construct bottom-up variance vector
			//mBottomUpVariance = new float[bottomupsize+1];
			//construct bottom-up age vector
			//bottomUpAge = new short[bottomupsize+1];
			/* for(int i = 0; i < bottomupsize; i++){
			        bottomUpAge[i] = 0;
			   }
			 */
			//construct the top-down mask vector
			//mTopDownMask = new float[topdownsize];
			/*for (int i = 0; i < topdownsize; i++) {
				mTopDownMask[i] = 1f;
			}*/
			//construct the top-down variance vector
			//mTopDownVariance = new float[topdownsize];
			//construct the top-down age vector
			//topDownAge = new short[topdownsize];
			/* for(int j = 0; j < topdownsize; j++){
			        topDownAge[j] = 0;
			 }
			 */
			//construct the lateral mask vector
//			mLateralMask = new float[lateralsize];
			//construct the lateral variance vector
//			mLateralVariance = new float[lateralsize];
			//construct the lateral age vector
//			lateralAge = new int[lateralsize];


			neiIndex = new int[lateralsize];
			for (int i = 0; i < lateralsize; i++) {
				neiIndex[i] = 0;
			}

			isGrowLateral = false;
		}

		// Make the neuron is look at the volume dimension index
		//setBottomUpMask(1.0f, mBottomUpMask.length-1);
	}

	//construct neuron
	/**
	 * This constructor will initialize a neuron instance with an unknown neuron type and unknown input size..
	 * Mostly used in topKCompetition and DynamictopKCompetition.
	 *
	 * @param bottomupsize
	 * @param topdownsize
	 * @param lateralsize
	 * @param category
	 * @param index
	 */
	public Neuron(int bottomupsize, int topdownsize, int lateralsize, boolean category, int index) {
		//set the neuron's type
		mType = 0;
		//set the neuron's firing age
		mFiringAge = 0;
		/*
		 * newResponse=0.0f; oldResponse=0.0f;
		 */
		//set the neuron's index
		mIndex = index;
		//set the neuron's category: 0 is z neuron; 1 is y neuron
		mCategory = category;
		//set the neuron's winner flag
		winnerFlag = false;
		//set machine zero
		MACHINE_FLOAT_ZERO = 0.00001f;
		//construct the neuron's 3D location vector
		mLocation = new float[3];
		//construct each neuron's bottom-up weight vector
		mBottomUpWeights = new float[bottomupsize+2]; // +2 for serotonin and dopamine in motor neuron
		//construct each neuron's top-down weight vector
		mTopDownWeights = new float[topdownsize];
		//construct each neuron's lateral weight vector
		mLateralWeight =  new ArrayList<Pair> ();
		//initialize z neuron's lateral connection
		if (mCategory == false && lateralsize > 0){
			//construct z neuron's lateral mask vector
			mLateralMasks =  new ArrayList<Integer> ();
			//construct z neuron's lateral age vector
			//lateralAges = new ArrayList<Integer> ();
			//initialize z neuron's lateral mask
/*			for (int i = 0; i < lateralsize; i++) {
				mLateralMasks.add(i);
			}	*/
			//construct z neuron's lateral variance vector
			//mLateralVariance = new float[lateralsize];
			for (int i = 0; i < lateralsize; i++) {
//				mLateralMask[i] = 1.0f;
				mLateralMasks.add(i);
				//lateralAges.add(0);
				Pair a = new Pair(i,0);
				mLateralWeight.add(a);
			}
		}
		//initialize y neuron's weights
		if (mCategory) {
			MACHINE_FLOAT_ZERO = 0.0001f;
			//randomly initialize each neuron's bottom-up weight vector
			initializeWeight(mBottomUpWeights, bottomupsize);
			//randomly initialize each neuron's top-down weights
			initializeWeight(mTopDownWeights, topdownsize);
			//construct the bottom-up mask vector
			//mBottomUpMask = new float[bottomupsize];
			//construct the bottom-up variance vector
			//mBottomUpVariance = new float[bottomupsize];
			//construct the bottom-up age vector
			//bottomUpAge = new short[bottomupsize];
			//initialize the bottom-up ages
//			for (int i = 0; i < bottomupsize; i++) {
//				bottomUpAge[i] = 0;
//			}
			//construct the top-down mask vector
			//mTopDownMask = new float[topdownsize];
			//construct the top-down variance vector
			//mTopDownVariance = new float[topdownsize];
			//construct the top-down age vector
			//topDownAge = new short[topdownsize];
			//initialize the top-down ages
//			for (int j = 0; j < topdownsize; j++) {
//				topDownAge[j] = 0;
//			}
			//lateralAges = new ArrayList<Integer>();
			//construct the lateral mask vector
			mLateralMasks =  new ArrayList<Integer> ();
			lateralGrowlist = new ArrayList<Integer>();
			lateralDeletelist = new ArrayList<Integer>();
			//construct the lateral variance vector
			//mLateralVariance = new float[lateralsize];
			//construct the lateral age vector
			//lateralAges =  new ArrayList<Integer> ();
			float[] lateralWeights = new float[mLateralSize];
			initializeWeight(lateralWeights, lateralsize);
			for(int i=0; i<mLateralSize; i++){
				mLateralMasks.add(i);
				//lateralAges.add(0);
				Pair a = new Pair(i,lateralWeights[i]);
				mLateralWeight.add(a);
			}
		}


		neiIndex = new int[lateralsize];
		for (int i = 0; i < lateralsize; i++) {
			neiIndex[i] = 0;
		}

		isGrowLateral = false;

	}

	// seed for the random number generator
	protected static long seed = 0; // System.currentTimeMillis();
	Random rand = new Random(seed);

	/**
	 * Initialize the weights for the neuron with random float values.
	 *
	 * @param weights Array to be initialized with random values
	 * @param size Number of elements in the array
	 */
	private void initializeWeight(float[] weights, int size) {
		for (int i = 0; i < size; i++) {
			weights[i] = rand.nextFloat();
		}
		seed = rand.nextInt();
	}

	/**
	 * Element-wise multiplication between two 1D vectors of the same size.
	 *
	 * @param vec1 First vector for multiplication
	 * @param vec2 Second vector for multiplication
	 *
	 * @return 1D array result from the multiplication between vec1 and vec2
	 */
	public float[] elementWiseProduct(float[] vec1, float[] vec2) {
		//check the length of 2 vectors
		assert vec1.length == vec2.length;
		int size = vec1.length;
		//construct the result vector
		float[] result = new float[vec2.length];
		//calculate the product between each element of 2 vectors
		for (int i = 0; i < size; i++) {
			result[i] = vec1[i] * vec2[i];
		}
		return result;
	}

	/**
	 * This method returns a static array of integer values from an ArrayList<Integer>.
	 *
	 * @param a Arraylist<Integer> with the dynamic weights of the neurons
	 * @return A static array of int values used for the DN updates.
	 */
	private int[] getIntArrayList(ArrayList<Integer> a){
		int[] b = new int[a.size()];
		for(int i=0; i<a.size(); i++){
			b[i] = a.get(i);
		}
		return b;
	}

	/**
	 * This method returns a static array of float values from an ArrayList<Float>.
	 *
	 * @param a Arraylist<Float> with the dynamic weights of the neurons
	 * @return A static array of float values used for the DN updates.
	 */
	private float[] getFloatArrayList(ArrayList<Float> a){
		float[] b = new float[a.size()];
		for(int i=0; i<a.size(); i++){
			b[i] = a.get(i);
		}
		return b;
	}

	/**
	 * This method extract values for each Pair instances and add them to a static array of float values
	 *
	 * @param a Arraylist<Integer> with the dynamic weights of the neurons
	 * @return A static array of int values used for the DN updates
	 */
	private float[] getPairArrayList(ArrayList<Pair> a){
		float[] b = new float[a.size()];
		for(int i=0; i<a.size(); i++){
			b[i] = a.get(i).value;
		}
		return b;
	}

	/**
	 * This method updates the float values to each elements from the ArrayList<Pair>.
	 *
	 * @param a List of instances with index and value
	 * @param b Array of the updated float values
	 */
	private void instorePairArrayList(ArrayList<Pair> a, float[] b){
		for(int i = 0; i < a.size(); i++){
			Pair ab = new Pair(a.get(i).index, b[i]);
			a.set(i, ab);
		}
	}

	/**
	 * This method increments the neuron's firing age if it is the winner.
	 */
	public void addFiringAge() {
		if(winnerFlag) {
			mFiringAge++;
		}
	}

	//hebbian learning for each neuron
	/**
	 * This method does the hebbian learning computation of the hidden neurons.
	 *
	 * @param sensorInput All the SensorLayers' neuron responses
	 * @param motorInput All the MotorLayers' neuron responses
	 * @param lateralResponse All the Lateral neuron responses
	 */
	public void hebbianLearnHidden(float[] sensorInput, float[] motorInput, float[] lateralResponse) {
		float bottomUpMeanVariance = 0;
		float topDownMeanVariance = 0;
		float lateralMeanVariance = 0;
		if (winnerFlag == true) {
			if(DN2.DEBUG) {
				System.out.println("update neuron id: " + this.getindex());
			}
			//increase the firing age
			mFiringAge++;
			//update the bottom-up connections
			if (mType == 4 || mType == 5 || mType == 6 || mType == 7) {
				//filter the bottom-up input vector through bottom-up mask
				float[] currentSensorInput = sensorInput;
				//currentSensorInput = epsilon_normalize(currentSensorInput, currentSensorInput.length, true);
				//increase bottom-up synapses' ages
				//incrementBottomUpAge();
				//update the bottom-up weights
				updateWeights(mBottomUpWeights, currentSensorInput, getAmnesicLearningRate(mFiringAge),
						false);
				//accumulate the variance between the bottom-up weight and input vector
				//float[] currBottomUpVar = calculateDiff(mBottomUpWeights, currentSensorInput, mBottomUpMask);
				//updateWeights(mBottomUpVariance, mBottomUpMask, currBottomUpVar,
				//		getAmnesicLearningRate(bottomUpAge, mBottomUpMask), false);
				//if the neuron reach the stable stage, start bottom-up synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					bottomUpMeanVariance = mean(mBottomUpVariance, mBottomUpMask);
//					if (bottomUpMeanVariance < 0.01f) {
//						bottomUpMeanVariance = 0.01f;
//					}
//					//update the bottom-up mask
//					updateBottomUpMask(bottomUpMeanVariance, currentSensorInput);
//				}
				/*System.out.println("");
				int sze = sensorInput.length;
                Bitmap bitmapSensor = Bitmap.createBitmap(45, 2 * 45, Bitmap.Config.ARGB_8888);
				//ByteBuffer bf = ByteBuffer.allocate(4*sze / 3);
				int x = 0;
				int y = 0;
				for (int i = 0; i < sze; i += 3) {
					int R = (int) sensorInput[i] & 0xFF;// << 24;
					int G = (int) sensorInput[i + 1] & 0xFF;// << 16;
					int B = (int) sensorInput[i + 2] & 0xFF;// << 8;
					//int A = 255; //(int) sensorInput[i + 3];
                    bitmapSensor.setPixel(x, y, Color.rgb(R, G, B));
                    x++;
                    if(x%45 == 0) {
                        y++;
                        x = 0;
                    }
					//bf.putInt(A + R + G + B);
				}

				///bf.position(0);
				//bitmapSensor.copyPixelsFromBuffer(bf);

				long time = System.currentTimeMillis();
				String sensorPath = StorageUtils.currentTrainingImgDir + File.separator + "t:" + String.valueOf(time)
						+ "_NeuronInput.png";
				StorageUtils.SaveBitmap(bitmapSensor, sensorPath);

				//ByteBuffer bf2 = ByteBuffer.allocate(4*sze / 3);
                Bitmap bitmapWeights = Bitmap.createBitmap(45, 2 * 45, Bitmap.Config.ARGB_8888);
				x=0;
				y=0;
				for (int i = 0; i < sze; i += 3) {
					int R = (int) (mBottomUpWeights[i] * mNorm + mMean) & 0xFF;// << 24;
					int G = (int) (mBottomUpWeights[i + 1] * mNorm + mMean) & 0xFF;// << 16;
					int B = (int) (mBottomUpWeights[i + 2] * mNorm + mMean) & 0xFF;// << 8;
					int A = (int) 255;//(int) (mBottomUpWeights[i + 3] * mNorm + mMean);
                    bitmapWeights.setPixel(x, y, Color.rgb(R, G, B));
                    x++;
                    if(x%45 == 0) {
                        y++;
                        x = 0;
                    }
					//bf2.putInt(A + R + G + B);
				}

				//bf2.position(0);
				//bitmapWeights.copyPixelsFromBuffer(bf2);

				String savePath = StorageUtils.currentTrainingImgDir + File.separator + "t:" + String.valueOf(time)
						+ "Age:" + mFiringAge + "Index:" + mIndex + "_BtmUpWeights.png";
				StorageUtils.SaveBitmap(bitmapWeights, savePath);

				System.out.println("");*/
			}
			//update the top-down connections
			if (mType == 1 || mType == 3 || mType == 5 || mType == 7) {
				//filter the top-down input vector through top-down mask
				float[] currentMotorInput = motorInput;
				//normalize the top-down input vector
				//currentMotorInput = epsilon_normalize(currentMotorInput, currentMotorInput.length, false);
				//increase top-down synapses' ages
				//incrementTopDownAge();
				//update the top-down weights
				updateWeights(mTopDownWeights, currentMotorInput, getAmnesicLearningRate(mFiringAge),
						false);
				//accumulate the variance between the top-down weight and input vector
				//float[] currTopDownVar = calculateDiff(mTopDownWeights, currentMotorInput, mTopDownMask);
				//updateWeights(mTopDownVariance, mTopDownMask, currTopDownVar,
				//		getAmnesicLearningRate(topDownAge, mTopDownMask), false);
				//if the neuron reach the stable stage, start top-down synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					topDownMeanVariance = mean(mTopDownVariance, mTopDownMask);
//					if (topDownMeanVariance < 0.01f) {
//						topDownMeanVariance = 0.01f;
//					}
//					//update the top-down mask
//					updateTopDownMask(topDownMeanVariance);
//				}
			}
			//update the lateral connection
			if (mType == 2 || mType == 3 || mType == 6 || mType == 7) {
				float[] currentlateralInput;
				//filter the top-down input vector through lateral mask
				currentlateralInput = getCurrentInput(lateralResponse, mLateralMasks);
				//normalize the lateral input vector
				//currentlateralInput = epsilon_normalize(currentlateralInput);
//              currentlateralInput =
//              normalize(currentlateralInput,currentlateralInput.length,2);

				//increase lateral synapses' ages
				//incrementLateralAge();
				//update the lateral weights
				float[] lateralWeights = getPairArrayList(mLateralWeight);
//				int[] curAges = getIntArrayList(lateralAges);
//				short[] ages = new short[curAges.length];
//				for(int i = 0; i < curAges.length; i++)
//				{
//					ages[i] = (short)curAges[i];
//				}
				updateWeights(lateralWeights, currentlateralInput, getLearningRate(mFiringAge), false);
				instorePairArrayList(mLateralWeight,lateralWeights);
				if(mType == 3){
					System.out.println("weights size: "+mLateralWeight.size());
				}
				//accumulate the variance between the lateral weight and input vector
				//float[] currLateralVar = calculateDiff(mLateralWeight, currentlateralInput, mLateralMasks);
				//updateWeights(mLateralVariance, mLateralMasks, currLateralVar, getAmnesicLearningRate(ages), false);
				//if the neuron reach the stable stage, start lateral synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					lateralMeanVariance = mean(mLateralVariance, mLateralMasks);
//					if (lateralMeanVariance < 0.01f) {
//						lateralMeanVariance = 0.01f;
//					}
//					//update the lateral mask
//					updateLateralMask(lateralMeanVariance);
//				}
			}
			if(mFiringAge > SMAGE){
				mFiringAge = 0;
			}

			// To update the motor neurons dopamine and serotonin levels
			updateSerotoninLevel();
			updateDopamineLevel();
		}
	}

	//hebbian learning for each neuron
	public void hebbianLearnHiddenParallel(float[][] sensorInput, float[] motorInput, float[] lateralResponse) {
		float bottomUpMeanVariance = 0;
		float topDownMeanVariance = 0;
		float lateralMeanVariance = 0;
		if (winnerFlag == true) {
			System.out.println("update neuron id: " + this.getindex());
			//increase the firing age
			mFiringAge++;
			//update the bottom-up connections
			if (mType == 4 || mType == 5 || mType == 6 || mType == 7) {
				//increase bottom-up synapses' ages
				//incrementBottomUpAge();
				float[][] currentSensorInput = new float[sensorInput.length][];
				float[][] currentWeight = new float[sensorInput.length][];
				float[][] currentMask = new float[sensorInput.length][];
				float[][] currentVariance = new float[sensorInput.length][];
				short[][] currentAge = new short[sensorInput.length][];
				int beginIndex = 0;
				for(int i=0; i<sensorInput.length; i++){
					currentSensorInput[i] = new float[sensorInput[i].length];
					currentWeight[i] = new float[sensorInput[i].length];
					currentMask[i] = new float[sensorInput[i].length];
					currentVariance[i] = new float[sensorInput[i].length];
					currentAge[i] = new short[sensorInput[i].length];
					System.arraycopy(mBottomUpWeights, beginIndex, currentWeight[i], 0, sensorInput[i].length);
					//System.arraycopy(mBottomUpMask, beginIndex, currentMask[i], 0, sensorInput[i].length);
					//System.arraycopy(mBottomUpVariance, beginIndex, currentVariance[i], 0, sensorInput[i].length);
					//System.arraycopy(bottomUpAge, beginIndex, currentAge[i], 0, sensorInput[i].length);
					beginIndex += sensorInput[i].length;
					//filter the bottom-up input vector through bottom-up mask
					currentSensorInput[i] = elementWiseProduct(sensorInput[i], sign(currentMask[i]));
					currentSensorInput[i] = epsilon_normalize(currentSensorInput[i], currentSensorInput[i].length, true);
					//update the bottom-up weights
					updateWeights(currentWeight[i], currentSensorInput[i], getAmnesicLearningRate(currentAge[i]),false);
					//accumulate the variance between the bottom-up weight and input vector
					float[] currBottomUpVar = calculateDiff(currentWeight[i], currentSensorInput[i], currentMask[i]);
//					updateWeights(currentVariance[i], currentMask[i], currBottomUpVar,
//						getAmnesicLearningRate(currentAge[i], currentMask[i]), false);
				}
				int begin = 0;
				float[] currentSensorInputs = new float [beginIndex];
				for(int i=0; i<sensorInput.length; i++){
					System.arraycopy(currentWeight[i], 0, mBottomUpWeights, begin, sensorInput[i].length);
					//System.arraycopy(currentMask[i], 0, mBottomUpMask, begin, sensorInput[i].length);
					//System.arraycopy(currentVariance[i], 0, mBottomUpVariance, begin, sensorInput[i].length);
					//System.arraycopy(currentAge[i], 0, bottomUpAge, begin, sensorInput[i].length);
					System.arraycopy(currentSensorInput[i], 0, currentSensorInputs, begin, sensorInput[i].length);
					begin += sensorInput[i].length;
				}
				//if the neuron reach the stable stage, start bottom-up synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					bottomUpMeanVariance = mean(mBottomUpVariance, mBottomUpMask);
//					if (bottomUpMeanVariance < 0.01f) {
//						bottomUpMeanVariance = 0.01f;
//					}
				//update the bottom-up mask
//				updateBottomUpMask(bottomUpMeanVariance, currentSensorInputs);
//				}

			}
			//update the top-down connections
			if (mType == 1 || mType == 3 || mType == 5 || mType == 7) {
				//filter the top-down input vector through top-down mask
				float[] currentMotorInput = motorInput;
				//normalize the top-down input vector
				currentMotorInput = epsilon_normalize(currentMotorInput, currentMotorInput.length, false);
				//increase top-down synapses' ages
				//incrementTopDownAge();
				//update the top-down weights
				updateWeights(mTopDownWeights, currentMotorInput, getAmnesicLearningRate(mFiringAge),
						false);
//				//accumulate the variance between the top-down weight and input vector
//				float[] currTopDownVar = calculateDiff(mTopDownWeights, currentMotorInput, mTopDownMask);
////				updateWeights(mTopDownVariance, mTopDownMask, currTopDownVar,
////						getAmnesicLearningRate(topDownAge, mTopDownMask), false);
//				//if the neuron reach the stable stage, start top-down synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					topDownMeanVariance = mean(mTopDownVariance, mTopDownMask);
//					if (topDownMeanVariance < 0.01f) {
//						topDownMeanVariance = 0.01f;
//					}
//					//update the top-down mask
//					updateTopDownMask(topDownMeanVariance);
//				}
			}
			//update the lateral connection
			if (mType == 2 || mType == 3 || mType == 6 || mType == 7) {
				float[] currentlateralInput;
				//filter the top-down input vector through lateral mask
				currentlateralInput = getCurrentInput(lateralResponse, mLateralMasks);
				//normalize the lateral input vector
				currentlateralInput = epsilon_normalize(currentlateralInput);
//	              currentlateralInput =
//	              normalize(currentlateralInput,currentlateralInput.length,2);

				//increase lateral synapses' ages
				//incrementLateralAge();
				//update the lateral weights
				float[] lateralWeights = getPairArrayList(mLateralWeight);
//				int[] curAges = getIntArrayList(lateralAges);
//				short[] ages = new short[curAges.length];
//				for(int i = 0; i < curAges.length; i++)
//				{
//					ages[i] = (short)curAges[i];
//				}
				updateWeights(lateralWeights, currentlateralInput, getLearningRate(mFiringAge), false);
				instorePairArrayList(mLateralWeight,lateralWeights);
				if(mType == 3){
					System.out.println("weights size: "+mLateralWeight.size());
				}
//				//accumulate the variance between the lateral weight and input vector
//				float[] currLateralVar = calculateDiff(mLateralWeight, currentlateralInput, mLateralMasks);
//				updateWeights(mLateralVariance, mLateralMasks, currLateralVar, getAmnesicLearningRate(ages), false);
//				//if the neuron reach the stable stage, start lateral synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					lateralMeanVariance = mean(mLateralVariance, mLateralMasks);
//					if (lateralMeanVariance < 0.01f) {
//						lateralMeanVariance = 0.01f;
//					}
//					//update the lateral mask
//					updateLateralMask(lateralMeanVariance);
//				}
			}
			if(mFiringAge > SMAGE){
				mFiringAge = 0;
			}
		}
	}


	/**
	 * Hebbian Learning for the MotorLayer neurons with input from HiddenLayer neurons (Y -> Z).
	 *
	 * @param sensorInput Response array from all the HiddenLayers' neurons.
	 */
	public void hebbianLearnHidden(float[] sensorInput) {
		//increase the firing age
		mFiringAge++;
		//normalize the input vector
		normalize1(sensorInput, sensorInput.length, 2);

		//normalize the weight vector
/*		if(mFiringAge % bottomupFrequency == 0){
		    normalize1(mBottomUpWeights, mBottomUpWeights.length, 2);
		}*/

		//update bottom-up weights
		updateWeights(mBottomUpWeights, sensorInput, getLearningRate(mFiringAge));

		// To update the motor neurons dopamine and serotonin levels
		updateSerotoninLevel();
		updateDopamineLevel();
	}

	/**
	 * Hebbian Learning from the MotorLayer neurons with input from HiddenLayer neurons (Z -> Z').
	 *
	 * @param currentlateralInput Response array of all the MotorLayer neurons connected to the current neuron.
	 */
	public void hebbianLearnLateral(float[] currentlateralInput) {
		//increase the z neuron's lateral ages
		//incrementLateralAge();
		//normalize the lateral input
		currentlateralInput = normalize(currentlateralInput, currentlateralInput.length, 2);
		//int[] curAges = getIntArrayList(lateralAges);
		float[] lateralWeights = getPairArrayList(mLateralWeight);
		//update the lateral weight
//		short[] ages = new short[curAges.length];
//		for(int i = 0; i < curAges.length; i++)
//		{
//			ages[i] = (short)curAges[i];
//		}
		updateWeights(lateralWeights, currentlateralInput, getLearningRate(mFiringAge), true);
		instorePairArrayList(mLateralWeight,lateralWeights);

	}

	//hebbian learning for each neuron
	/**
	 * Hebbian Learning for the MotorLayer neurons with input from the HiddenLayer neurons (Y -> Z).
	 *
	 */
	public void PrimaryhebbianLearnHidden(float[] Input) {
		float bottomUpMeanVariance = 0;
		float topDownMeanVariance = 0;
		float lateralMeanVariance = 0;
		if (winnerFlag == true) {
			if(mState == true){
				//increase the firing age
				mFiringAge++;
			}
			else{
				mFiringAge = 1;
			}
			if(mType == 4){
				//update the bottom-up connections
				//filter the bottom-up input vector through bottom-up mask
				//float[] currentSensorInput = elementWiseProduct(Input, sign(mBottomUpMask));
				//normalize the bottom-up input vector
				//currentSensorInput = normalize(currentSensorInput, Input.length, 2);
				//increase bottom-up synapses' ages
//				incrementBottomUpAge();
				//update the bottom-up weights
				//updateWeights(mBottomUpWeights, currentSensorInput, getAmnesicLearningRate((short)mFiringAge),true);
			}
			if(mType == 1){
				float[] currentMotorInput = Input;
				//normalize the top-down input vector
				currentMotorInput = epsilon_normalize(currentMotorInput, currentMotorInput.length, false);
				//increase top-down synapses' ages
//				incrementTopDownAge();
				//update the top-down weights
				updateWeights(mTopDownWeights, currentMotorInput, getAmnesicLearningRate((short)mFiringAge),true);
			}


		}
	}

	// Need to remove?
	private float ratioParameter(float ratio) {
		float Ratio;
		if (ratio < SMLOWERTHRESH) {
			Ratio = 1;

		} else if (ratio > SMUPPERTHRESH) {
			Ratio = 0;
		} else {
			Ratio = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);
		}
		return Ratio;
	}

	/**
	 * binary conditioning of each element in the vector:
	 * <ul>
	 * 		<li> 1 for the ones larger than 0 <\li>
	 * 		<li> 0 for the others <\li>
	 *
	 * @param fs Array of float numbers needed to do binary conditioning.
	 * @return Array of float numbers with values of zero or one.
	 */
	protected float[] sign(float[] fs) {
		//construct the result rector
		float[] result = new float[fs.length];
		for (int i = 0; i < fs.length; i++) {
			result[i] = (fs[i] > 0) ? 1 : 0;
		}
		return result;
	}

	/**
	 * This method takes a global list of neurons and a local list of connected neurons b.
	 * The list b contain the index id of the neurons connected with the current neuron.
	 * Return an array c of the responses of b neurons stored in a.
	 * Dense representation, no holes between neurons.
	 *
	 * @param a Static array of float values responses.
	 * @param b Dynamic array of Integer indexes of neurons connected with the current Neuron instance.
	 * @return Static array of neuron responses connected with the current Neuron instance
	 */
	private float[] getCurrentInput(float[] a, ArrayList<Integer> b){
		float[] c = new float[b.size()];
		for(int i=0; i<b.size(); i++){
			c[i] = a[b.get(i)];
		}
		return c;
	}

	/**
	 * This method normalizes the non-zero elements in the weight array. Only normalizes the values where the mask array has a value of 1.
	 *
	 * @param weight Array of weights
	 * @param length Number of neurons connected to the current Neuron instance.
	 * @return
	 */
	protected float[] epsilon_normalize(float[] weight, int length, boolean isBottomUp) {
		float norm = 0;
		float mean = 0;
		int useful_length = 0;
		for (int i = 0; i < length; i++) {
			//		if (mask[i] > 0) {
			mean += weight[i];
			useful_length++;

		}
		//calculate the mean value of non-zero elements in the vector
		mean = mean / useful_length - MACHINE_FLOAT_ZERO;
		boolean nor = false;
		if(length > 50 && mean > 1) {
			mMean = mean;
			nor = true;
		}
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
		norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
		if(length > 50)// && nor)
			mNorm = norm;
		//normalize the non-zero element
		if (norm > 0) {
			for (int i = 0; i < length; i++) {
				weight[i] = weight[i] / norm;
			}
		}

//		// Volume Dimension Procedure
//		if(isBottomUp) {
//			if(norm > maxVolume) {
//				maxVolume = norm;
//			}
//			// TODO: Currently running: VD without volumeContribution
//			weight[weight.length-1] = volumeContribution * (maxVolume - norm) / maxVolume;
//			//weight[weight.length-1] = (maxVolume - norm) / maxVolume;
//			norm = maxVolume / ((1-volumeContribution) * norm);
//			//norm = maxVolume / ( norm);
//			//norm /= (1- volumeContribution);
//
//
//			//if(length > 50)// && nor)
//			//mNorm = norm;
//			//normalize the non-zero element
//			if (norm > 0) {
//				for (int i = 0; i < length; i++) {
//					weight[i] = weight[i] / norm;
//				}
//			}
//
//
//			norm = 0;
//			for (int i = 0; i < length; i++) {
//				//if (mask[i] > 0) {
//					norm += weight[i] * weight[i];
//
//			}
//			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
//			if (norm > 0) {
//				for (int i = 0; i < length; i++) {
//					weight[i] = weight[i] / norm;
//				}
//			}
//
//		}
		return weight;
	}

	/**
	 * This method normalize all the non-zero elements in the vector.
	 *
	 * @param weight Vector of elements
	 * @return The normalized vector.
	 */
	static float[] epsilon_normalize(float[] weight) {
		float norm = 0;
		float mean = 0;
		for (int i = 0; i < weight.length; i++) {
			mean += weight[i];
		}
		//calculate the mean value of non-zero elements in the vector
		mean = mean / weight.length - MACHINE_FLOAT_ZERO;
		//each non-zero element minuses the mean value
		for (int i = 0; i < weight.length; i++) {
			weight[i] -= mean;
		}
		//calculate the norm-2 for the non-zero elements
		for (int i = 0; i < weight.length; i++) {
			norm += weight[i] * weight[i];
		}
		norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
		//normalize the non-zero element
		if (norm > 0) {
			for (int i = 0; i < weight.length; i++) {
				weight[i] = weight[i] / norm;
			}
		}
		return weight;
	}

//	// TODO Ask Xiang: What is the meanVariance?
//	/**
//	 * Update the bottom-up mask vector.
//	 *
//	 * @param meanVariance
//	 * @param currentSensorInput The response of all the SensorNeurons (X) in the network.
//	 */
//	protected void updateBottomUpMask(float meanVariance, float[] currentSensorInput) {
//		if (meanVariance > MACHINE_FLOAT_ZERO) {
//			//construct the grow array list
//			ArrayList<Integer> growlist = new ArrayList<Integer>();
//			//construct the cut array list
//			ArrayList<Integer> cutlist = new ArrayList<Integer>();
//
//			for (int j = 0; j < mBottomUpWeights.length; j++) {
//				//calculate the ratio for each variance element
//				float ratio = mBottomUpVariance[j] / meanVariance;
//				if (ratio < SMLOWERTHRESH && mBottomUpMask[j] > 0) {
//					mBottomUpMask[j] = 1;
//					// Grow nearby (neighbor of neuron i to z neuron j).
//					if (bottomUpAge[j] > SMAGE) {
//						growlist.add(j);
//					}
//				} else if (ratio > SMUPPERTHRESH && mBottomUpMask[j] > 0) {
//					// Cut connection.
//					if (bottomUpAge[j] > SMAGE) {
//						cutlist.add(j);
//					}
//				} else {
//					if (bottomUpAge[j] > SMAGE && mBottomUpMask[j] > 0) {
//						// Version 1: Mahalanobis distance.
//						mBottomUpMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);
//
//						/* Version 2: Linear distance.
//						   bottomUpMask[i][j] = (SMUPPERTHRESH - ratio)/
//						   (SMUPPERTHRESH - SMLOWERTHRESH);
//						   */
//					}
//				}
//			}
//
//			for (int j = 0; j < cutlist.size(); j++) {
//				mBottomUpMask[cutlist.get(j)] = 0;
//				mBottomUpWeights[cutlist.get(j)] = 0;
//				bottomUpAge[cutlist.get(j)] = 0;
//				//mBottomUpVariance[cutlist.get(j)] = 0;
//			}
//
//			/*  for (int j = 0; j < growlist.size(); j++) {
//			        growBottomUpConnection(i, growlist.get(j), currentSensorInput);
//			    } */
//
//			//mFiringAge = 0;
//			/* for (int j = 0; j < bottomUpWeights[i].length; j++){
//			       bottomUpAge[i][j] = 0;
//			       bottomUpVariance[i][j] = 0;
//			   }  */
//			//filter the bottom-up weights through new bottom-up mask
//			mBottomUpWeights = elementWiseProduct(mBottomUpWeights, sign(mBottomUpMask));
//			//normalize the bottom-up weight
//			mBottomUpWeights = normalize(mBottomUpWeights, mBottomUpWeights.length, 2);
//		}
//	}

	// TODO Ask Xiang: What is the meanVariance?
	/**
	 * Update the top-down mask vector.
	 *
	 * @param meanVariance
	 */
	protected void updateTopDownMask(float meanVariance) {
		if (meanVariance > MACHINE_FLOAT_ZERO) {
			//construct the grow array list
			ArrayList<Integer> growlist = new ArrayList<Integer>();
			//construct the cut array list
			ArrayList<Integer> cutlist = new ArrayList<Integer>();

//			for (int j = 0; j < mTopDownWeights.length; j++) {
//				//calculate the ratio for each variance element
//				float ratio = mTopDownVariance[j] / meanVariance;
//				if (ratio < SMLOWERTHRESH && mTopDownMask[j] > 0) {
//					mTopDownMask[j] = 1;
//					// Grow nearby (neighbor of neuron i to z neuron j).
//					if (topDownAge[j] > SMAGE) {
//						growlist.add(j);
//					}
//				} else if (ratio > SMUPPERTHRESH && mTopDownMask[j] > 0) {
//					// Cut connection.
//					if (topDownAge[j] > SMAGE) {
//						cutlist.add(j);
//					}
//				} else {
//					if (topDownAge[j] > SMAGE && mTopDownMask[j] > 0) {
//						// Version 1: Mahalanobis distance.
//						mTopDownMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) / (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);
//
//						// Version 2: Linear distance.
//						/* topDownMask[i][j] = (SMUPPERTHRESH - ratio)
//						   (SMUPPERTHRESH - SMLOWERTHRESH);  */
//					}
//				}
//			}

//			for (int j = 0; j < growlist.size(); j++) {
//				//growTopDownConnection(growlist.get(j));
//			}

			for (int j = 0; j < cutlist.size(); j++) {
				//mTopDownMask[cutlist.get(j)] = 0;
				mTopDownWeights[cutlist.get(j)] = 0;
				//topDownAge[cutlist.get(j)] = 0;
				//mTopDownVariance[cutlist.get(j)] = 0;
			}
			//normalize the top-down weight vector
			mTopDownWeights = epsilon_normalize(mTopDownWeights, mTopDownWeights.length, false);
		}
	}

	/*
	 * private void updateLateralMask(float meanVariance, float[]
	 * currentSensorInput) { if (meanVariance > MACHINE_FLOAT_ZERO) {
	 * ArrayList<Integer> growlist = new ArrayList<Integer>();
	 * ArrayList<Integer> cutlist = new ArrayList<Integer>();
	 *
	 * for (int j = 0; j < mLateralExcitationWeights.length; j++) { float ratio
	 * = mLateralVariance[j] / meanVariance; if (ratio < SMLOWERTHRESH &&
	 * mLateralMask[j] > 0) { mLateralMask[j] = 1; // Grow nearby (neighbor of
	 * neuron i to z neuron j). if (lateralAge[j] > SMAGE) { growlist.add(j); }
	 * } else if (ratio > SMUPPERTHRESH && mLateralMask[j] > 0) { // Cut
	 * connection. if (lateralAge[j] > SMAGE) { cutlist.add(j); } } else { if
	 * (lateralAge[j] > SMAGE && mLateralMask[j] > 0) { // Version 1:
	 * Mahalanobis distance. mLateralMask[j] = (1 / ratio - 1 / SMUPPERTHRESH) /
	 * (1 / SMLOWERTHRESH - 1 / SMUPPERTHRESH);
	 *
	 * // Version 2: Linear distance. // bottomUpMask[i][j] = (SMUPPERTHRESH -
	 * ratio)/ // (SMUPPERTHRESH - SMLOWERTHRESH); } } }
	 *
	 * for (int j = 0; j < cutlist.size(); j++) { mLateralMask[cutlist.get(j)] =
	 * 0; mLateralExcitationWeights[cutlist.get(j)] = 0;
	 * lateralAge[cutlist.get(j)] = 0; mLateralVariance[cutlist.get(j)] = 0; }
	 *
	 * //for (int j = 0; j < growlist.size(); j++) { //
	 * growBottomUpConnection(i, growlist.get(j), currentSensorInput); //}
	 *
	 * // mFiringAge = 0; // for (int j = 0; j < bottomUpWeights[i].length;
	 * j++){ // bottomUpAge[i][j] = 0; // bottomUpVariance[i][j] = 0; // }
	 * mLateralExcitationWeights = elementWiseProduct(mLateralExcitationWeights,
	 * sign(mLateralMask)); mLateralExcitationWeights =
	 * normalize(mLateralExcitationWeights, mLateralExcitationWeights.length,
	 * 2); } }
	 */

	// TODO Ask Xiang: What is the meanVariance?
	/**
	 * Update the lateral mask vector.
	 *
	 * @param meanVariance

	 */
//	private void updateLateralMask(float meanVariance) {
//		if (meanVariance > MACHINE_FLOAT_ZERO) {
//
//				for (int j = 0; j < mLateralMasks.size(); j++) {
//					//calculate the ratio for each variance element
//					//float ratio = mLateralVariance[mLateralMasks.get(j)] / meanVariance;
//					if (ratio < SMLOWERTHRESH) {
//						// Grow nearby (neighbor of neuron i to z neuron j).
//						if (lateralAges.get(j) > SMAGE) {
//							isGrowLateral = true;
//							lateralGrowlist.add(mLateralMasks.get(j));
//
//						}
//					}
//					else if (ratio > SMUPPERTHRESH) {
//						// Cut connection.
//						if (lateralAges.get(j) > SMAGE) {
//							lateralDeletelist.add(mLateralMasks.get(j));
//							mLateralMasks.remove(j);
//							mLateralWeight.remove(j);
//							lateralAges.remove(j);
//							j--;
//						}
//					}
///*					else {
//						if (lateralAges.get(j) <= SMAGE) {
//							// Version 1: Mahalanobis distance.
//							mLateralMasks.remove(j);
//							mLateralWeight.remove(j);
//							lateralAges.remove(j);
//							j--;
//						}
//
//					}*/
//				}
//
//
//            //update the lateral weight vector
//			float[] lateralWeights = getPairArrayList(mLateralWeight);
//			lateralWeights = epsilon_normalize(lateralWeights);
//			instorePairArrayList(mLateralWeight,lateralWeights);
//		}
//	}

//	/**
//	 * This method increases the top-down connections ages by one.
//	 */
//	private void incrementTopDownAge() {
//		//increase the top-down weight age which in the mask
////		for (int j = 0; j < mTopDownMask.length; j++) {
////			if (mTopDownMask[j] > 0) {
////				//topDownAge[j]++;
////			}
////		}
//	}

	//increase the lateral connections ages
/*	private void incrementLateralAge() {
		//increase the lateral weight age which in the mask
		for (int j = 0; j < mLateralMask.length; j++) {
			if (mLateralMask[j] > 0) {
				lateralAge[j]++;
			}
		}
	}
*/

//	/**
//	 * This method increases the lateral connections ages by one.
//	 */
//	private void incrementLateralAge() {
//		//increase the lateral weight age which in the mask
//		for (int j = 0; j < mLateralMasks.size(); j++) {
//			lateralAges.set(j, lateralAges.get(j)+1);
//		}
//	}


//	/**
//	 * This method increases the bottom-up connections ages by one.
//	 */
//	private void incrementBottomUpAge() {
//		//increase the bottom-up weight age which in the mask
//		for (int j = 0; j < bottomUpAge.length; j++) {
//			if (bottomUpAge[j] < Short.MAX_VALUE ) {
//
//				bottomUpAge[j]++;
//			}
//		}
//	}

	/*
	 * // TODO: implement this. private void growBottomUpConnection(int j,
	 * float[] currentSensorInput) { int[] sub = ind2sub(j, inputSize); // ind:
	 * [height, width] if (sub[0] - 1 >= 0) { int[] growSub = { sub[0] - 1,
	 * sub[1] }; int growInd = sub2ind(growSub, inputSize); if
	 * (mBottomUpMask[growInd] == 0) { mBottomUpMask[growInd] = 1;
	 * mBottomUpWeights[growInd] = currentSensorInput[growInd];
	 * bottomUpAge[growInd] = 0; mBottomUpVariance[growInd] = 0; } } if (sub[0]
	 * + 1 < inputSize[0]) { int[] growSub = { sub[0] + 1, sub[1] }; int growInd
	 * = sub2ind(growSub, inputSize); if (mBottomUpMask[growInd] == 0) {
	 * mBottomUpMask[growInd] = 1; mBottomUpWeights[growInd] =
	 * currentSensorInput[growInd]; bottomUpAge[growInd] = 0;
	 * mBottomUpVariance[growInd] = 0; } } if (sub[1] - 1 >= 0) { int[] growSub
	 * = { sub[0], sub[1] - 1 }; int growInd = sub2ind(growSub, inputSize); if
	 * (mBottomUpMask[growInd] == 0) { mBottomUpMask[growInd] = 1;
	 * mBottomUpWeights[growInd] = currentSensorInput[growInd];
	 * bottomUpAge[growInd] = 0; mBottomUpVariance[growInd] = 0; } } if (sub[1]
	 * + 1 < inputSize[1]) { int[] growSub = { sub[0], sub[1] + 1 }; int growInd
	 * = sub2ind(growSub, inputSize); if (mBottomUpMask[growInd] == 0) {
	 * mBottomUpMask[growInd] = 1; mBottomUpWeights[growInd] =
	 * currentSensorInput[growInd]; bottomUpAge[growInd] = 0;
	 * mBottomUpVariance[growInd] = 0; } } }
	 */

	/*
	private int[] ind2sub(int i, int[] shape) {
		int[] result = new int[2];
		result[0] = (int) i / shape[1];
		result[1] = i - result[0] * shape[1];
		return result;
	}

	private int sub2ind(int[] ind, int[] shape) {
		int result = ind[1] + ind[0] * shape[1];
		return result;
	}
	*/

//	/**
//	 * Add the new top-down connections for each neuron
//	 * @param j
//	 */
//	private void growTopDownConnection(int j) {
//		//add the neuron's j-th (if j is larger than 0) top-down connection's neighbor (j-1)-th in the connections
//		if (j > 0) {
//			if (mTopDownMask[j - 1] == 0) {
//				mTopDownMask[j - 1] = 1;
//				//set the new connection's age
//				//topDownAge[j - 1] = 0;
//				mTopDownVariance[j - 1] = 0;
//				mTopDownWeights[j - 1] = 0;
//			}
//		}
//		//add the neuron's j-th (if j is less than length of the vector) top-down connection's neighbor (j+1)-th in the connections
//		if (j < mTopDownMask.length - 1) {
//			if (mTopDownMask[j + 1] == 0) {
//				mTopDownMask[j + 1] = 1;
//				//set the new connection's age
//				//topDownAge[j + 1] = 0;
//				mTopDownVariance[j + 1] = 0;
//				mTopDownWeights[j + 1] = 0;
//			}
//		}
//	}

/*	//add new lateral connections for each neuron
	private void growLateralConnection(int j, int num) {
		for(int k = 0; k < num; k++) {
			int index = neiIndex[k];
			boolean g = false;
			int t = 0;
			//add the neuron's neighbor lateral connection's neighbor (j-1)-th in the connections
			if (InlateralMask(index) == false) {
				g = true;
				mLateralMasks.add(index);
				//set the new connection's age
				lateralAges.add(0);
				mLateralVariance[index] = 0;
				Pair a = new Pair(index, 0);
				mLateralWeight.add(a);
			}

			while(g) {
				if(t <1) {
					System.out.println("excuted!");
					t += 1;
					}
			}
		}
	}*/


	/**
	 * Checks whether a neuron index i is present in the LateralMask.
	 *
	 * @param i Neuron index.
	 * @return Boolean value that indicates if the lateral neuron exists.
	 */
	public boolean InlateralMask(int i){
		boolean exist = false;
		for(int j=0; j<mLateralMasks.size(); j++){
			if(mLateralMasks.get(j) == i){
				exist = true;
			}
		}
		return exist;
	}

	/**
	 * Checks whether a neuron index i is present in the LateralDeleteList.
	 *
	 * @param i Neuron index.
	 * @return Boolean value that indicates if the lateral neuron exists.
	 */
	public boolean InlateralDeletelist(int i){
		boolean exist = false;
		for(int j=0; j<lateralDeletelist.size(); j++){
			if(lateralDeletelist.get(j) == i){
				exist = true;
			}
		}
		return exist;
	}

	/**
	 * Checks whether a neuron index i is present in the LateralMask.
	 *
	 * @param i Neuron index.
	 * @return Index of the lateral neuron.
	 */
	private int getlateralMaskIndex(int i){
		int exist = -1;
		for(int j=0; j<mLateralMasks.size(); j++){
			if(mLateralMasks.get(j) == i){
				exist = j;
			}
		}
		return exist;
	}


	/**
	 * This method takes two arrays and compute the difference of their elements, only if the mask value is greater than zero.
	 *
	 * Difference between array_1 and array_2
	 * 				  0            -> If mask is zero.
	 *      abs(array_1 - array_2) -> otherwise
	 *
	 *
	 * @param input_1 Float array of values
	 * @param input_2 Float array of values
	 * @param mask Integer array to determine
	 * @return result Float array of the computed differences
	 */
	protected float[] calculateDiff(float[] input_1, float[] input_2, float[] mask) {
		//check the 2 vectors have same length
		assert (input_1.length == input_2.length);
		//construct the result vector
		float[] result = new float[input_1.length];
		for (int i = 0; i < input_1.length; i++) {
			//calculate the difference if the element is in the mask
			if (mask[i] > 0) {
				result[i] = (Math.abs(input_1[i] - input_2[i]));
			} else {
				result[i] = 0;
			}
		}
		return result;
	}

	/**
	 * This method takes two arrays and compute the difference of their elements, only if the mask value is greater than zero.
	 *
	 * Difference between array_1 and array_2
	 * 				  0            -> If mask is zero.
	 *      abs(array_1 - array_2) -> otherwise
	 *
	 *
	 * @param input_1 Array of Pair
	 * @param input_2 Array of Pair
	 * @param mask Array of Integer
	 * @return result Float array of the computed differences
	 */
	protected float[] calculateDiff(ArrayList<Pair> input_1, float[] input_2, ArrayList<Integer> mask) {
		//check the 2 vectors have same length
		assert (input_1.size() == input_2.length);
		//construct the result vector
		float[] result = new float[input_2.length];
		for (int i = 0; i < input_2.length; i++) {
			result[i] = 0;
		}
		for (int i = 0; i < mask.size(); i++) {
			//calculate the difference if the element is in the mask
			result[i] = (Math.abs(input_1.get(i).value - input_2[i]));
		}
		return result;
	}

	/**
	 * Update the weight vector
	 *
	 * @param weights
	 * @param mask
	 * @param input
	 * @param learningRate
	 * @param normalize_flag
	 */
	protected void updateWeights(float[] weights, float[] mask, float[] input, float[] learningRate,
								 boolean normalize_flag) {
		// make sure both arrays have the same length
		assert weights.length == input.length;
		//update the element in the weight vector if the element is in the mask
		for (int i = 0; i < input.length; i++) {
			if (mask[i] > 0) {
				weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * newResponse * input[i];
			} else {
				weights[i] = 0;
			}
		}
		//normalize the weight vector if needed
		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

	/**
	 * update the weight vector
	 *
	 * @param weights
	 * @param mask
	 * @param input
	 * @param learningRate
	 * @param normalize_flag
	 */
	protected void updateWeights(float[] weights, ArrayList<Integer> mask, float[] input, float[] learningRate,
								 boolean normalize_flag) {
		// make sure both arrays have the same length
		assert mask.size() == input.length;
		//update the element in the weight vector if the element is in the mask
		for (int i = 0; i < mask.size(); i++) {
			weights[mask.get(i)] = (1.0f - learningRate[i]) * weights[mask.get(i)] + learningRate[i] * newResponse * input[i];
		}
		//normalize the weight vector if needed
		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}


	/**
	 * update the weight vector
	 *
	 * @param weights
	 * @param input
	 * @param learningRate
	 * @param normalize_flag
	 */
	protected void updateWeights(float[] weights, float[] input, float[] learningRate, boolean normalize_flag) {
		// make sure both arrays have the same length
//         assert weights.length == input.length;
		//update the element in the weight vector
		for (int i = 0; i < input.length; i++) {
			if(Float.isNaN(weights[i])){
				System.out.println("neuron meets nan: "+i);
			}
			// TODO: AK: Fix this equation so that it uses the neuron's scaled response to the input
			weights[i] = (1.0f - learningRate[i]) * weights[i] + learningRate[i] * newResponse * input[i]; // AK: Added newResponse to equation

		}

		//normalize the weight vector if needed
		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

	/**
	 * update the weight vector
	 *
	 * @param weights
	 * @param input
	 * @param learningRate
	 * @param normalize_flag
	 */
	private void updateWeights(float[] weights, float[] input, float learningRate, boolean normalize_flag) {
		// make sure both arrays have the same length
		assert weights.length == input.length;
		// For hidden neuron serotonin and dopamine increase the learning rate
		learningRate *= 1+serotoninLevel+dopamineLevel;  // Updated per eq 9.8 on page 329 of the green book
		//update the element in the weight vector
		for (int i = 0; i < input.length; i++) {
			weights[i] = (1.0f - learningRate) * weights[i] + learningRate * newResponse * input[i];
		}

		//normalize the weight vector if needed
		if (normalize_flag) {
			weights = normalize(weights, input.length, 2);
		}
	}

//	/**
//	 * This method calculates an array of learning rates.
//	 *
//	 * @param age
//	 * @param mask
//	 * @return
//	 */
//	protected float[] getAmnesicLearningRate(short[] age) {
//		//construct the result vector
//		float[] result = new float[age.length];
//		//calculate the learning rate for each element in the mask
//		for (int i = 0; i < age.length; i++) {
//				assert (age[i] > 0);
//				result[i] = getAmnesicLearningRate(age[i]);
//		}
//		return result;
//	}

	/**
	 * This method calculates an array of learning rates.
	 *
	 * @param age
	 * @return
	 */
	private float[] getAmnesicLearningRate(short[] age) {
		//construct the result vector
		float[] result = new float[age.length];
		//calculate the learning rate for each element
		for (int i = 0; i < age.length; i++) {
			assert (age[i] > 0);
			result[i] = getAmnesicLearningRate(age[i]);

		}
		return result;
	}

	/**
	 * This method calculates the amnesic learning rate of a single age.
	 *
	 * @param age
	 * @return
	 */
	private float getAmnesicLearningRate(short age) {
		//initialize the parameter mu and the result learning_rate
		float mu, learning_rate;
		//calculate mu according age
		if (age < T1) {
			mu = 0.0f;
		}
		else if ((age < T2) && (age >= T1)) {
			mu = C * ((float) age - T1) / (T2 - T1);
		}
		else {
			mu = C * ((float) age - T2) / GAMMA;
		}
		//calculate the learning rate
		learning_rate = (1 + mu) / ((float) age);

		return learning_rate;
	}

	private float getModulationAmnesicLearningRate(int age) {
		//initialize the parameter mu and the result learning_rate
		float mu, learning_rate;
		//calculate mu according age
		if (age < T1_mod) {
			mu = 0.0f;
		}
		else if ((age < T2_mod) && (age >= T1_mod)) {
			mu = C * ((float) age - T1_mod) / (T2_mod - T1_mod);
		}
		else {
			mu = C * ((float) age - T2_mod) / GAMMA_mod;
		}
		//calculate the learning rate
		learning_rate = (1 + mu) / ((float) age);

		return learning_rate;
	}

	/**
	 * This method calculates the learning rate of a single age.
	 *
	 * @param age
	 * @return
	 */
	private float getLearningRate(int age) {

		return (1.0f / ((float) age));
	}

	/**
	 * calculate the bottom-up preResponse.
	 *
	 * @param sensorInput
	 */
	public float computeBottomUpResponse(float[] sensorInput) {
		bottomUpResponse = 0.0f;
//      assert sensorInput.length==mBottomUpWeights.length;

		//mPainInput = (sensorInput[sensorInput.length-8] + sensorInput[sensorInput.length-7] + sensorInput[sensorInput.length-6] + sensorInput[sensorInput.length-5]) / 4;
		//mSweetInput = (sensorInput[sensorInput.length-4] + sensorInput[sensorInput.length-3] + sensorInput[sensorInput.length-2] + sensorInput[sensorInput.length-1]) / 4;

		//calculate for the y neuron
		if (this.mCategory) {

			// Set the serotonin and dopamine input for the neuron. Levels only updated when neuron updates
//			float temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
//			temp1 = sensorInput[sensorInput.length-8];
//			temp2 = sensorInput[sensorInput.length-7];
//			temp3 = sensorInput[sensorInput.length-6];
//			temp4 = sensorInput[sensorInput.length-5];
//			temp5 = sensorInput[sensorInput.length-4];
//			temp6 = sensorInput[sensorInput.length-3];
//			temp7 = sensorInput[sensorInput.length-2];
//			temp8 = sensorInput[sensorInput.length-1];
//
//			mPainInput = (temp1 + temp2 + temp3 + temp4) / 4;
//			mSweetInput = (temp5 + temp6 + temp7 + temp8) / 4;

			mPainInput = 0;
			mSweetInput = 0;
			for (int i = 1; i <= numMotor; i ++){
				mPainInput += sensorInput[sensorInput.length - numMotor - i];
				mSweetInput += sensorInput[sensorInput.length - i];
			}
			mPainInput /= numMotor;
			mSweetInput /= numMotor;

			//old static way to compute
//			mPainInput = (sensorInput[sensorInput.length-8] + sensorInput[sensorInput.length-7] + sensorInput[sensorInput.length-6] + sensorInput[sensorInput.length-5]) / 4;
//			mSweetInput = (sensorInput[sensorInput.length-4] + sensorInput[sensorInput.length-3] + sensorInput[sensorInput.length-2] + sensorInput[sensorInput.length-1]) / 4;

			if (mType == 4 || mType == 5 || mType == 6 || mType == 7) {
				float[] currentWeight = new float[mBottomUpWeights.length];
				//copy the current weight vector
				System.arraycopy(mBottomUpWeights, 0, currentWeight, 0, mBottomUpWeights.length);
				//filer the input vector through mask
				//sensorInput = elementWiseProduct(sensorInput, sign(mBottomUpMask));
				//sensorInput = epsilon_normalize(sensorInput, sensorInput.length, true);
				//filter the weight vector through mask
				//currentWeight = elementWiseProduct(currentWeight, sign(mBottomUpMask));
				currentWeight = epsilon_normalize(currentWeight, mBottomUpWeights.length, false);
				//calculate the response
				bottomUpResponse += dotProduct(currentWeight, sensorInput, mBottomUpWeights.length);
				// System.out.println("The bottom-up "+bottomUpResponse);
			}
		} else {

			// Set the serotonin and dopamine input for the neuron. Levels only updated when neuron updates
			mPainInput = sensorInput[sensorInput.length-2];
			mSweetInput = sensorInput[sensorInput.length-1];

			//added to normalize response
			float[] currentWeight = new float[mBottomUpWeights.length];
			float[] currentSensor = new float[sensorInput.length];
			System.arraycopy(mBottomUpWeights, 0, currentWeight, 0, mBottomUpWeights.length);
			currentWeight = epsilon_normalize(currentWeight, mBottomUpWeights.length, true); //if change to normalize, use flag 2 (others should be deleted)
			System.arraycopy(sensorInput, 0, currentSensor, 0, sensorInput.length);
			currentSensor = epsilon_normalize(currentSensor, currentSensor.length, true);
			bottomUpResponse += dotProduct(currentWeight, currentSensor, mBottomUpWeights.length);

			if (Float.isNaN(bottomUpResponse)){
				System.out.println("the " +mIndex +" bottom up is nan");
			}

//			//normalize the weight vector
//			normalize1(mBottomUpWeights, mBottomUpWeights.length, 2);
//			//calculate response
//			bottomUpResponse = dotProduct(mBottomUpWeights, sensorInput, mBottomUpWeights.length);
		}
		return bottomUpResponse;
	}

	/**
	 * calculate the bottom-up preResponse
	 * @param sensorInput
	 */
	public void computeBottomUpResponseInParallel(float[][] sensorInput) {
		bottomUpResponse = 0.0f;
//      assert sensorInput.length==mBottomUpWeights.length;
		//calculate for the y neuron

		if (mType == 4 || mType == 5 || mType == 6 || mType == 7) {
			float[][] currentWeight = new float[sensorInput.length][];
			float[][] currentMask = new float[sensorInput.length][];
			int beginIndex = 0;
			//copy the current weight vector
			for (int j = 0; j < sensorInput.length; j++) {
				currentWeight[j] = new float[sensorInput[j].length];
				currentMask[j] = new float[sensorInput[j].length];
				System.arraycopy(mBottomUpWeights, beginIndex, currentWeight[j], 0, sensorInput[j].length);
				//System.arraycopy(mBottomUpMask, beginIndex, currentMask[j], 0, sensorInput[j].length);
				beginIndex += sensorInput[j].length;
				//filer the input vector through mask
				sensorInput[j] = elementWiseProduct(sensorInput[j], sign(currentMask[j]));
				sensorInput[j] = epsilon_normalize(sensorInput[j], sensorInput[j].length, true);
				//filter the weight vector through mask
				currentWeight[j] = elementWiseProduct(currentWeight[j], sign(currentMask[j]));
				currentWeight[j] = epsilon_normalize(currentWeight[j], sensorInput[j].length, false);
				//calculate the response
				bottomUpResponse += dotProduct(currentWeight[j], sensorInput[j], sensorInput[j].length);
			}
			bottomUpResponse = bottomUpResponse/sensorInput.length;
			// System.out.println("The bottom-up "+bottomUpResponse);
		}

	}

	/**
	 * calculate the top-down preResponse
	 *
	 * @param motorInput
	 */
	public float computeTopDownResponse(float[] motorInput) {
		topDownResponse = 0.0f;
		if ((mType == 1 || mType == 3 || mType == 5 || mType == 7) || !mCategory) {
			float[] currentWeight = new float[mTopDownWeights.length];
//          assert motorInput.length==mTopDownWeights.length;
			//copy the current weight vector
			System.arraycopy(mTopDownWeights, 0, currentWeight, 0, mTopDownWeights.length);
			//filer the input vector through mask
			//motorInput = elementWiseProduct(motorInput, sign(mTopDownMask));
			//normalize the input vector
			//motorInput = epsilon_normalize(motorInput, mTopDownWeights.length, false);
			//normalize the weight vector
			currentWeight = epsilon_normalize(currentWeight, mTopDownWeights.length, false);
//			motorInput = normalize(motorInput, mTopDownWeights.length, 2);
//			currentWeight = normalize(currentWeight, mTopDownWeights.length, 2);
			//calculate response
			topDownResponse += dotProduct(currentWeight, motorInput, mTopDownWeights.length);
		}
		return topDownResponse;
	}

	/**
	 * calculate the lateral preResponse
	 * @param response
	 * @param lateralpercent
	 */
	public float computeLateralResponse(float[] response, float lateralpercent) {
		lateralResponse = 0;
		if (mType == 2 || mType == 3 || mType == 6 || mType == 7 || mCategory == false) {
			//filer the input vector through mask
			float[] currentInput = getCurrentInput(response, mLateralMasks);
			float[] currentWeight = getPairArrayList(mLateralWeight);
//			if(mType==3){
//                System.out.println("The neuron "+mIndex+"'s lateral length: "+currentInput.length+" "+currentWeight.length);
//            }
			//normalize the input vector
			currentInput = normalize(currentInput, currentInput.length, 2);
			//normalize the weight vector
			currentWeight = normalize(currentWeight, currentInput.length, 2);
			//calculate response
			lateralResponse += lateralpercent * dotProduct(currentWeight, currentInput, mLateralWeight.size());
//          System.out.println("The lateral "+lateralResponse);
/*			if(mType==3){
                System.out.println("The neuron "+mIndex+"'s lateral "+lateralResponse);
            }*/

		}
		return lateralResponse;
	}


	float mTopDown = 0.5f; // Default Top Down

	public void setTopDownWeights(float topDown){
		mTopDown = topDown;
	}

	public void setSweetInput(float input) {mSweetInput = input;}
	public void setPainInput(float input) {mPainInput = input;}

	/**
	 * calculate the total preResponse
	 */
	public float computeResponse(boolean isMotor) {
		//calculate the total preResponse for different type y neurons

		if (this.mCategory) {

			bottomUpResponse *= DNHandler.computeResponseWeights[0];
			lateralResponse *= DNHandler.computeResponseWeights[1];
			topDownResponse *= DNHandler.computeResponseWeights[2];

			if (mType == 7) {
				newResponse = (bottomUpResponse + topDownResponse + lateralResponse) /
						((DNHandler.computeResponseWeights[0] + DNHandler.computeResponseWeights[1] + DNHandler.computeResponseWeights[2]));
			}
			if (mType == 6) {
				newResponse = (bottomUpResponse + lateralResponse) / ( (DNHandler.computeResponseWeights[0] + DNHandler.computeResponseWeights[1]));
			}
			if (mType == 5) {
				newResponse = (bottomUpResponse + topDownResponse) / ( (DNHandler.computeResponseWeights[0] + DNHandler.computeResponseWeights[2]));
				//System.out.println(newResponse);
			}
			if (mType == 4) {
				newResponse = bottomUpResponse / DNHandler.computeResponseWeights[0];
			}
			if (mType == 3) {
				newResponse = (lateralResponse + topDownResponse) / ((DNHandler.computeResponseWeights[1] + DNHandler.computeResponseWeights[2]));
			}
			if (mType == 2) {
				newResponse = lateralResponse / DNHandler.computeResponseWeights[1];
			}
			if (mType == 1) {
				newResponse = topDownResponse / DNHandler.computeResponseWeights[2];
			}
		}
		//calculate the total preResponse for z neurons
		else {
			// TODO: Check to see if lateral for motor includes other concept zones. Should it be this way or not?
			// AK: Changed this to weight lateral response less than bottom up for motor neurons
			//newResponse = (bottomUpResponse + lateralResponse) / 2.0f;
			if(Float.isNaN(bottomUpResponse)){
				System.out.println("the "+mIndex+" Bottom Up bp response meets nan");
			}

			if(Float.isNaN(lateralResponse)){
				System.out.println("the "+mIndex+" Lateral bp response meets nan");
			}

			newResponse = (bottomUpResponse + lateralResponse) / (DNHandler.computeMotorResponseWeights[0] + DNHandler.computeMotorResponseWeights[1]);
			if(Float.isNaN(newResponse)){
				System.out.println("the "+mIndex+" motor bp response meets nan");
			}
		}

		// TODO: Add bool parameter to only do this for motor neurons
		if (isMotor) {
			// Equation 9.6 on page 320 of the green book:
//			float motivatedResponse = newResponse * (1 - 0.3f * serotoninLevel + 0.1f * dopamineLevel);
			float motivatedResponse = newResponse * (1 - serotoninMultiplier * serotoninLevel + dopamineMultiplier * dopamineLevel);
			if (motivatedResponse < 0) motivatedResponse = 0;
			newResponse = motivatedResponse;
		}

		// TODO: Uncomment this to add in serotonin and dopamine flushing
		// AK: Decrease the serotonin and dopamine amounts in the neuron
		/*float learningRate = getModulationAmnesicLearningRate(this.mFiringAge);
		serotoninLevel = (1.0f - learningRate) * serotoninLevel;
		dopamineLevel = (1.0f - learningRate) * dopamineLevel;*/

//		serotoninLevel = (1.0f - 0.2f) * serotoninLevel;
//		dopamineLevel = (1.0f - 0.2f) * dopamineLevel;

		return newResponse;
//      System.out.println("response: "+newResponse);
	}

	/**
	 * transfer the new response value to the old response
	 */
	public void replaceResponse() {
		oldResponse = newResponse;
	}

	/**
	 * do inner product between 2 vectors
	 * @param a
	 * @param b
	 * @param size
	 * @return
	 */
	protected float dotProduct(float[] a, float[] b, int size) {
		float r = 0.0f;

		for (int i = 0; i < size; i++) {
			r += (a[i] * b[i]);
		}

		return r;
	}

	/**
	 * multiply a constant for each element in a vector
	 * @param constant
	 * @param array
	 */
	private void multilyConstant(float constant, float[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = array[i] * constant;
		}
	}

	void updateSerotoninLevel() {
		float serotoninInput = mPainInput; // painsensor.getinput
		float learningRate = getModulationAmnesicLearningRate(this.mFiringAge);
		serotoninLevel = (1.0f - learningRate) * serotoninLevel + learningRate * serotoninInput * newResponse; // TODO: Use newResponse to punish neurons relative to response
		if (Float.isNaN(serotoninLevel)){
			System.out.println("Serotonin Nan");
		}
	}

	void updateDopamineLevel() {
		float dopamineInput = mSweetInput;
		float learningRate = getModulationAmnesicLearningRate(this.mFiringAge);
		dopamineLevel = (1.0f - learningRate) * dopamineLevel + learningRate * dopamineInput * newResponse;
		if (Float.isNaN(dopamineLevel)){
			System.out.println("Dopamine Nan");
		}
	}

	public float getSerotoninLevel() {
		return serotoninLevel;
	}

	public float getDopamineLevel() {
		return dopamineLevel;
	}

	/**
	 * set the neuron's type
	 * @param type
	 */
	public void setType(int type) {
		mType = type;
	}

	/**
	 * get the neuron's type
	 * @return
	 */
	public int getType() {
		return mType;
	}

	/**
	 * set the neuron's winner flag value
	 * @param flag
	 */
	public void setwinnerflag(boolean flag) {
		winnerFlag = flag;
	}

	/**
	 * get the neuron's winner flag value
	 * @return
	 */
	public boolean getwinnerflag() {
		return winnerFlag;
	}

	/**
	 * set old response
	 * @param response
	 */
	public void setoldresponse(float response) {
		oldResponse = response;
	}

	/**
	 * get the old response
	 * @return
	 */
	public float getoldresponse() {
		return oldResponse;
	}

	/**
	 * set the new response
	 * @param response
	 */
	public void setnewresponse(float response) {
		newResponse = response;
	}

	/**
	 * get the new response
	 * @return
	 */
	public float getnewresponse() {
		return newResponse;
	}

	/**
	 * set the bottom-up reResponse
	 * @param response
	 */
	public void setbottomUpresponse(float response) {
		bottomUpResponse = response;
	}

	/**
	 * get the bottom-up reResponse
	 * @return
	 */
	public float getbottomUpresponse() {
		return bottomUpResponse;
	}

	/**
	 * set the top-down preResponse
	 * @param response
	 */
	public void settopDownresponse(float response) {
		topDownResponse = response;
	}

	/**
	 * get the top-down preResponse
	 * @return
	 */
	public float gettopDownresponse() {
		return topDownResponse;
	}

	/**
	 * set the lateral preResponse
	 * @param response
	 */
	public void setlateralExcitationresponse(float response) {
		lateralResponse = response;
	}

	/**
	 * get the lateral preResponse
	 * @return
	 */
	public float getlateralExcitationresponse() {
		return lateralResponse;
	}

	/**
	 * set the firing age
	 * @param firingage
	 */
	public void setfiringage(short firingage) {
		mFiringAge = firingage;
	}

	/**
	 * get the firing age
	 * @return
	 */
	public short getfiringage() {
		return mFiringAge;
	}

//	/**
//	 * set the top-down synapses' ages
//	 * @param topdownages
//	 */
//	public void settopdownages(int[] topdownages) {
//		System.arraycopy(topdownages, 0, topDownAge, 0, topdownages.length);
//	}

//	/**
//	 * set one top-down synapse's age
//	 * @param topdownage
//	 * @param index
//	 */
//	public void settopdownage(short topdownage, int index) {
//		topDownAge[index] = topdownage;
//	}
//
//	/**
//	 * get the top-down synapses' ages
//	 * @return
//	 */
//	public short[] gettopdownage() {
//		return topDownAge;
//	}

//	/**
//	 * set the lateral synapses' ages
//	 * @param lateralages
//	 */
//	public void setlateralages(int[] lateralages) {
//        for(int i=0; i<lateralAges.size(); i++){
//        	lateralAges.set(mLateralMasks.get(i), lateralages[mLateralMasks.get(i)]);
//        }
//	}
//
//	/**
//	 * set one lateral synapse's age
//	 * @param lateralage
//	 * @param index
//	 */
//	public void setlateralage(int lateralage, int index) {
//		if(InlateralMask(index)){
//			int inx = getlateralMaskIndex(index);
//		     lateralAges.set(mLateralMasks.get(inx), lateralage);
//		}
//	}

//	/**
//	 * get the lateral synapses' ages
//	 * @return
//	 */
//	public int[] getlateralage() {
//		int[] lateralAge = new int[mLateralSize];
//		for(int i=0; i<mLateralMasks.size(); i++){
//			lateralAge[mLateralMasks.get(i)] = lateralAges.get(i);
//		}
//		return lateralAge;
//	}

//	/**
//	 * set the bottom-up synapses' ages
//	 * @param bottomupages
//	 */
//	public void setbottomupages(int[] bottomupages) {
//		System.arraycopy(bottomupages, 0, bottomUpAge, 0, bottomupages.length);
//	}
//
//	/**
//	 * set one bottom-up synapse's age
//	 * @param bottomupage
//	 * @param index
//	 */
//	public void setbottomupage(short bottomupage, int index) {
//		bottomUpAge[index] = bottomupage;
//	}
//
//	/**
//	 * get the bottom-up synapses' ages
//	 * @return
//	 */
//	public short[] getbottomupage() {
//		return bottomUpAge;
//	}

	/**
	 * set the neuron's index
	 * @param index
	 */
	public void setindex(int index) {
		mIndex = index;
	}

	/**
	 * get the neuron's index
	 * @return
	 */
	public int getindex() {
		return mIndex;
	}

	/**
	 * set the neuron's 3D location
	 * @param location
	 */
	public void setlocation(float[] location) {
		System.arraycopy(location, 0, mLocation, 0, location.length);

	}

	/**
	 * get the neuron's 3D location
	 * @return
	 */
	public float[] getlocation() {
		return mLocation;
	}

	/**
	 * get the top-down weight vector
	 * @return
	 */
	public float[] getTopDownWeights() {
		return mTopDownWeights;
	}

	/**
	 * set the top-down weight vector
	 * @param topDownWeights
	 */
	public void setTopDownWeights(float[] topDownWeights) {
		System.arraycopy(topDownWeights, 0, mTopDownWeights, 0, topDownWeights.length);

	}

	/**
	 * set one lateral weight element
	 * @param LateralWeight
	 * @param index
	 */
	public void setLateralWeight(float LateralWeight, int index) {
		if(InlateralMask(index)){
			int inx = getlateralMaskIndex(index);
			Pair lateralW = new Pair(mLateralMasks.get(inx), LateralWeight);
			mLateralWeight.set(mLateralMasks.get(inx), lateralW);
		}
	}

	/**
	 * get the lateral weight vector
	 * @return
	 */
	public float[] getLateralWeights() {
		float[] lateralWeights = new float[mLateralSize];
		for(int i=0; i<mLateralMasks.size(); i++){
			lateralWeights[mLateralMasks.get(i)] = mLateralWeight.get(i).value;
		}
		return lateralWeights;
	}

	/**
	 * set the lateral weight vector
	 * @param LateralWeights
	 */
	public void setLateralWeights(float[] LateralWeights) {
		for(int i=0; i<mLateralWeight.size(); i++){
			mLateralWeight.set(mLateralMasks.get(i), new Pair(mLateralMasks.get(i),LateralWeights[mLateralMasks.get(i)]));
		}
	}

	/**
	 * set one top-down weight element
	 * @param topDownWeight
	 * @param index
	 */
	public void setTopDownWeight(float topDownWeight, int index) {
		this.mTopDownWeights[index] = topDownWeight;
	}

	/**
	 * get the bottom-up weight vector
	 * @return
	 */
	public float[] getBottomUpWeights() {
		return mBottomUpWeights;
	}

	/**
	 * set the bottom-up weight vector
	 * @param bottomUpWeights
	 */
	public void setBottomUpWeights(float[] bottomUpWeights) {
		System.arraycopy(bottomUpWeights, 0, mBottomUpWeights, 0, bottomUpWeights.length);
	}

	/**
	 * set one bottom-up weight element
	 * @param bottomUpWeight
	 * @param index
	 */
	public void setBottomUpWeight(float bottomUpWeight, int index) {
		this.mBottomUpWeights[index] = bottomUpWeight;
	}

//	/**
//	 * get the top-down variance vector
//	 * @return
//	 */
//	public float[] getTopDownVariances() {
//		return mTopDownVariance;
//	}
//
//	/**
//	 * set the top-down variance vector
//	 * @param topDownVariance
//	 */
//	public void setTopDownVariances(float[] topDownVariance) {
//		System.arraycopy(topDownVariance, 0, mTopDownVariance, 0, topDownVariance.length);
//	}
//
//	/**
//	 * set one top-down variance vector element
//	 * @param topDownVariance
//	 * @param index
//	 */
//	public void setTopDownVariance(float topDownVariance, int index) {
//		this.mTopDownVariance[index] = topDownVariance;
//	}

//	/**
//	 * get the lateral variance vector
//	 * @return
//	 */
//	public float[] getLateralVariances() {
//		return mLateralVariance;
//	}
//
//	/**
//	 * set the lateral variance vector
//	 * @param lateralVariance
//	 */
//	public void setLateralVariances(float[] lateralVariance) {
//		System.arraycopy(lateralVariance, 0, mLateralVariance, 0, lateralVariance.length);
//
//	}
//
//	/**
//	 * set one lateral variance vector element
//	 * @param LateralVariance
//	 * @param index
//	 */
//	public void setLateralVariance(float LateralVariance, int index) {
//		this.mLateralVariance[index] = LateralVariance;
//	}

//	/**
//	 * get the bottom-up variance vector
//	 * @return
//	 */
//	public float[] getBottomUpVariances() {
//		return mBottomUpVariance;
//	}

//	/**
//	 * set the bottom-up variance vector
//	 * @param bottomUpVariance
//	 */
//	public void setBottomUpVariances(float[] bottomUpVariance) {
//		System.arraycopy(bottomUpVariance, 0, mBottomUpVariance, 0, bottomUpVariance.length);
//	}

//	/**
//	 * set one bottom-up variance vector element
//	 * @param bottomUpVariance
//	 * @param index
//	 */
//	public void setBottomUpVariance(float bottomUpVariance, int index) {
//		this.mBottomUpVariance[index] = bottomUpVariance;
//	}

//	/**
//	 * get the top-down mask vector
//	 * @return
//	 */
//	public float[] getTopDownMask() {
//		return mTopDownMask;
//	}
//
//	/**
//	 * set the top-down mask vector
//	 * @param topDownMask
//	 */
//	public void setTopDownMasks(float[] topDownMask) {
//		System.arraycopy(topDownMask, 0, mTopDownMask, 0, topDownMask.length);
//	}
//
//	/**
//	 * set one top-down mask vector element
//	 * @param topDownMask
//	 * @param index
//	 */
//	public void setTopDownMask(float topDownMask, int index) {
//		this.mTopDownMask[index] = topDownMask;
//	}

	/**
	 * get the lateral mask vector
	 * @return
	 */
	public float[] getLateralMask() {
		float[] lateralMask = new float[mLateralSize];
		for(int i=0; i<mLateralMasks.size(); i++){
			lateralMask[mLateralMasks.get(i)] = 1.0f;
		}
		return lateralMask;
	}

	/**
	 * set the lateral mask vector
	 * @param lateralMask
	 */
	public void setLateralMasks(float[] lateralMask) {
		for(int i=0; i<lateralMask.length; i++){
			if(InlateralMask(i) == false){
				mLateralMasks.add(i);
				mLateralWeight.add(new Pair(i,0));
				//lateralAges.add(0);
			}
		}
	}

	/**
	 * add one lateral mask vector element
	 * @param a
	 */
	public void addLateralMasks(int a) {

		if(InlateralMask(a) == false){
			mLateralMasks.add(a);
			mLateralWeight.add(new Pair(a,0));
			//lateralAges.add(0);
			//mLateralVariance[a] = 0;
		}

	}

	/**
	 * set one lateral mask vector element
	 * @param lateralMask
	 * @param index
	 */
	public void setLateralMask(float lateralMask, int index) {
		if(InlateralMask(index) == false){
			mLateralMasks.add(index);
			mLateralWeight.add(new Pair(index,0));
			//lateralAges.add(0);

		}
	}

//	/**
//	 * get the bottom-up mask vector
//	 * @return
//	 */
//	public float[] getBottomUpMask() {
//		return mBottomUpMask;
//	}

//	/**
//	 * set the bottom-up mask vector
//	 * @param bottomUpMask
//	 */
//	public void setBottomUpMasks(float[] bottomUpMask) {
//		System.arraycopy(bottomUpMask, 0, mBottomUpMask, 0, bottomUpMask.length);
//	}
//
//	/**
//	 * set one bottom-up mask vector element
//	 * @param bottomUpMask
//	 * @param index
//	 */
//	public void setBottomUpMask(float bottomUpMask, int index) {
//		this.mBottomUpMask[index] = bottomUpMask;
//	}

	/**
	 * normalize the vector
	 * @param input
	 * @param size
	 * @param flag
	 * @return
	 */
	public float[] normalize(float[] input, int size, int flag) {
		float[] weight = new float[size];
		System.arraycopy(input, 0, weight, 0, size);
		if (flag == 1) {
			float min = weight[0];
			float max = weight[0];
			//find the max element and minimum element in the vector
			for (int i = 0; i < size; i++) {
				if (weight[i] < min) {
					min = weight[i];
				}
				if (weight[i] > max) {
					max = weight[i];
				}
			}
			//calculate the difference between max and min value
			float diff = max - min + MACHINE_FLOAT_ZERO;
			//normalize the element form 0 to 1
			for (int i = 0; i < size; i++) {
				weight[i] = (weight[i] - min) / diff;
			}

			float mean = 0;
			for (int i = 0; i < size; i++) {
				mean += weight[i];
			}
			mean = mean / size;
			for (int i = 0; i < size; i++) {
				weight[i] = weight[i] - mean + MACHINE_FLOAT_ZERO;
			}

			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm);
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 2) {
			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 3) {
			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i];
			}
			norm = norm + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}
		return weight;
	}

	//update the weight vector
	protected void updateWeights(float[] weights, float[] input, float learningRate) {

		// make sure both arrays have the same length
		assert weights.length == input.length;

		for (int i = 0; i < input.length; i++) {
			weights[i] = (1.0f - learningRate) * weights[i] + learningRate * newResponse * input[i];
		}
	}

	//reset the responses
	public void resetResponses() {
		bottomUpResponse = 0.0f;
		topDownResponse = 0.0f;
		lateralResponse = 0.0f;
	}

	//calculate the mean value for a vector
	public static float mean(float[] m, float[] mask) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < m.length; i++) {
			if (mask[i] > 0) {
				sum += m[i];
				count++;
			}
		}
		return sum / count;
	}

	//calculate the mean value for a vector
	public static float mean(float[] m, ArrayList<Integer> mask) {
		float sum = 0;
		float count = 0;
		for (int i = 0; i < mask.size(); i++) {
			sum += m[mask.get(i)];
		}
		return sum / mask.size();
	}

	//normalize the vector
	public float[] normalize1(float[] weight, int size, int flag) {
		if (flag == 1) {
			float min = weight[0];
			float max = weight[0];
			//find the max element and minimum element in the vector
			for (int i = 0; i < size; i++) {
				if (weight[i] < min) {
					min = weight[i];
				}
				if (weight[i] > max) {
					max = weight[i];
				}
			}
			//calculate the difference between max and min value
			float diff = max - min + MACHINE_FLOAT_ZERO;
			//normalize the element form 0 to 1
			for (int i = 0; i < size; i++) {
				weight[i] = (weight[i] - min) / diff;
			}

			float mean = 0;
			for (int i = 0; i < size; i++) {
				mean += weight[i];
			}
			mean = mean / size;
			for (int i = 0; i < size; i++) {
				weight[i] = weight[i] - mean + MACHINE_FLOAT_ZERO;
			}

			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 2) {
			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i] * weight[i];
			}
			norm = (float) Math.sqrt(norm) + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}

		if (flag == 3) {
			float norm = 0;
			for (int i = 0; i < size; i++) {
				norm += weight[i];
			}
			norm = norm + MACHINE_FLOAT_ZERO;
			if (norm > 0) {
				for (int i = 0; i < size; i++) {
					weight[i] = weight[i] / norm;
				}
			}
		}
		return null;
	}

	//get the neuron's lateral response
	public float getlateralresponse() {
		return lateralResponse;
	}
	//set the neuron's state
	public void setState(boolean state){
		mState = state;
	}
	//get the neuron's state
	public boolean getState(){
		//return the neuron's state
		return mState;
	}

	private float sumVector(float[] a){
		float sum = 0;
		for(int i=0;i<a.length; i++){
			sum += a[i];
		}
		return sum;
	}

	private float sumArray(ArrayList<Pair> a){
		float sum = 0;
		for(int i=0;i<a.size(); i++){
			sum += a.get(i).value;
		}
		return sum;
	}


	//calculate the distance between the two 3-D locations
	public float computeDistance(float[] vector){
		//make sure the two location vector have same length
		assert vector.length == mLocation.length;
		//calculate the distance
		float delta_h = (mLocation[0]-vector[0])*(mLocation[0]-vector[0]);
		float delta_v = (mLocation[1]-vector[1])*(mLocation[1]-vector[1]);
		float delta_d = (mLocation[2]-vector[2])*(mLocation[2]-vector[2]);
		float dis = (float)Math.sqrt((double)(delta_h+delta_v+delta_d));
		return dis;

	}

	public void setneiIndex(int[] a) {

		for (int i = 0; i < neiIndex.length; i++) {
			neiIndex[i] = 0;
		}
		for (int i = 0; i < a.length; i++) {
			neiIndex[i] = a[i];
		}
	}

	public int[] getneiIndex() {
		return neiIndex;
	}

	public void setIsGrowLateral(boolean a) {
		isGrowLateral = a;
	}

	public boolean getIsGrowLateral() {
		return isGrowLateral;
	}

	public void setLateralGrowList(int[] a) {
		for(int i = 0; i < a.length; i++) {
			lateralGrowlist.add(a[i]);
		}
	}

	public void clearLateralGrowList() {
		int a =  lateralGrowlist.size()-1;
		while (a >= 0) {
			lateralGrowlist.remove(a);
			a -= 1;
		}
	}

	public void clearLateralDeleteList() {
		int a =  lateralDeletelist.size()-1;
		while (a >= 0) {
			lateralDeletelist.remove(a);
			a -= 1;
		}
	}

	public int[] getLateralGrowList() {
		int a =  lateralGrowlist.size();
		int[] b = new int[a];
		for(int i = 0; i < a; i++) {
			b[i] = lateralGrowlist.get(i);
		}
		return b;
	}

	// Nam added
	public void setSynapticMaintenanceAge(int age) {
		SMAGE = age;

	}

	public class Pair implements Comparable<Pair>, Serializable {
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
	/**
	 * Empty constructor - only used by RS_Neuron
	 */
	protected Neuron() {
//		System.out.println();
	}

	/**
	 * Sets necessary values - only used by RS_Neuron
	 *
	 * @param bottomupsize size of bottom up weights
	 * @param topdownsize  size of top down weights
	 * @param lateralsize  size of lateral weights
	 * @param category     1 for y neuron, 0 for z neuron
	 * @param type         Neuron type
	 * @param index        Number index of neuron
	 * @param inputsize    size of the input array
	 */
	protected void Setup(int bottomupsize, int topdownsize, int lateralsize, boolean category, int type, int index, int[] inputsize, int numMotor) {

		this.numMotor = numMotor;
		//set neuron's index
		mIndex = index;
		//set neuron's type: 1 for y neuron; 0 for z neuron
		mType = type;
		//initialize neuron's category; 1 for y neuron; 0 for z neuron
		mCategory = category;
		//initialize neuron's firing age
		mFiringAge = 0;
		//initialize neuron's winner flag
		winnerFlag = false;
		//initialize response
		newResponse = 0.0f;
		//initialize response
		oldResponse = 0.0f;
		//set the
		mLateralSize = lateralsize;
		//set the machine zero
		MACHINE_FLOAT_ZERO = 0.00001f;

		mBottomUpSize = bottomupsize + 1 + numMotor*2;
		mTopDownSize = topdownsize;

		mBottomUpWeights = null;
		//mBottomUpMask = null;
		//mBottomUpVariance = null;
		//bottomUpAge = null;

		//mTopDownMask = null;
		mTopDownWeights = null;
		//mTopDownVariance = null;
		//topDownAge = null;

		mLateralWeights = null;

		//set the input size
		inputSize = new int[inputsize.length];
		for (int i = 0; i < inputsize.length; i++) {
			inputSize[i] = inputsize[i];
		}
		//construct the 3D location vector
		mLocation = new float[3];

		//initialize the z neuron's lateral mask
		//mLateralWeight = new ArrayList<Pair>();
		//initialize the variances between lateral inputs and lateral weights
		//mLateralVariance = new float[lateralsize];

		// Initialize a Z neuron
		//if (mCategory == false && lateralsize > 0) {
		//initialize the lateral mask
//			mLateralMask = new float[lateralsize];
		//mLateralMasks = new ArrayList<Integer>();
		//lateralAges = new ArrayList<Integer>();
//				for (int i = 0; i < lateralsize; i++) {
////				mLateralMask[i] = 1.0f;
//					//mLateralMasks.add(i);
//					//lateralAges.add(0);
//					Pair a = new Pair(i, 0);
//					mLateralWeight.add(a);
//				}

		//initialize the lateral synapses' ages
//			lateralAge = new int[lateralsize];
		//}
		//initialize weights for Y neuron
//			if (mCategory) {
//				//set machine zero
//				MACHINE_FLOAT_ZERO = 0.0001f;
////				mLateralMasks = new ArrayList<Integer>();
////				//lateralAges = new ArrayList<Integer>();
////				lateralGrowlist = new ArrayList<Integer>();
////				lateralDeletelist = new ArrayList<Integer>();
//
//				//initialize y neuron's lateral weights for the neurons which have lateral connections
//				if (mType == 2 || mType == 3 || mType == 6 || mType == 7) {
////					float[] lateralWeights = new float[mLateralSize];
////					initializeWeight(lateralWeights, lateralsize);
////					for (int i = 0; i < mLateralSize; i++) {
////						mLateralMasks.add(i);
////						//lateralAges.add(0);
////						Pair a = new Pair(i, lateralWeights[i]);
////						mLateralWeight.add(a);
////					}
////				}
//				}
		//neiIndex = new int[lateralsize];

		//isGrowLateral = false;
		//}
	}

//	/**
//	 * Takes in allocations from RS_Neuron to copy to arrays of Neuron, necessary for
//	 * saving the DN
//	 *
//	 * @param bottomUpWeights  Allocation that holds the bottom up weights
//	 * @param topDownWeights   Allocation that holds the top down weights
//	 */
//	protected void LoadAllocationToArray(Allocation bottomUpWeights, Allocation topDownWeights, Allocation lateralWeights) {
//		mBottomUpWeights = new float[mBottomUpSize];
//		bottomUpWeights.copyTo(mBottomUpWeights);
//		bottomUpWeights.destroy();
//
//		mTopDownWeights = new float[mTopDownSize];
//		topDownWeights.copyTo(mTopDownWeights);
//		topDownWeights.destroy();
//
//		mLateralWeights = new float[mLateralSize];
//		lateralWeights.copyTo(mLateralWeights);
//		lateralWeights.destroy();
//
//
//	}
//
//	/**
//	 * Takes in RS_Neuron Allocations and copies Neuron arrays into them, necessary for loading DN
//	 *
//	 * @param bottomUpWeights  Allocation that will hold the bottom up weights
//	 * @param topDownWeights   Allocation that will hold the top down weights
//	 */
//	protected void reloadAllocations(Allocation bottomUpWeights, Allocation topDownWeights, Allocation lateralWeights) {
//		bottomUpWeights.copyFrom(mBottomUpWeights);
//		//bottomUpMask.copyFrom(mBottomUpMask);
//		//bottomUpVariance.copyFrom(mBottomUpVariance);
//		//bottomUpAgeAlloc.copyFrom(bottomUpAge);
//
//		mBottomUpWeights = null;
//		//mBottomUpMask = null;
//		//mBottomUpVariance = null;
//		//bottomUpAge = null;
//
//		topDownWeights.copyFrom(mTopDownWeights);
//		//topDownMask.copyFrom(mTopDownMask);
//		//topDownVariance.copyFrom(mTopDownVariance);
//		//topDownAgeAlloc.copyFrom(topDownAge);
//
//		mTopDownWeights = null;
//		//mTopDownMask = null;
//		//mTopDownVariance = null;
//		//topDownAge = null;
//
//		lateralWeights.copyFrom(mLateralWeights);
//		mLateralWeights = null;
//	}
//
//	public void reloadAllocations() {
//	}
//
//	public void reinitializeRenderScript(RenderScript rs) {
//	}

	/**
	 * Used by RS_Neuron because lateral hebbian learning is not done by
	 * RenderScript at this point
	 *
	 * @param lateralResponse array of lateral responses
	 */
	protected void lateralHebbianLearn(float[] lateralResponse) {

		if(winnerFlag) {
			float lateralMeanVariance = 0f;
			if (mType == 2 || mType == 3 || mType == 6 || mType == 7 || mCategory == false) {
				float[] currentlateralInput;
				//filter the top-down input vector through lateral mask
				//currentlateralInput = getCurrentInput(lateralResponse, mLateralMasks);
				//normalize the lateral input vector
				//currentlateralInput = epsilon_normalize(currentlateralInput);
//              currentlateralInput =
//              normalize(currentlateralInput,currentlateralInput.length,2);

				//increase lateral synapses' ages
				//incrementLateralAge();
				//update the lateral weights
				float[] lateralWeights = getPairArrayList(mLateralWeight);
//				int[] curAges = getIntArrayList(lateralAges);
//				short[] ages = new short[curAges.length];
//				for (int i = 0; i < curAges.length; i++) {
//					ages[i] = (short) curAges[i];
//				}
				updateWeights(lateralWeights, lateralResponse, getLearningRate(mFiringAge), false);

				instorePairArrayList(mLateralWeight, lateralWeights);
				if (mType == 3) {
					System.out.println("weights size: " + mLateralWeight.size());
				}
				//accumulate the variance between the lateral weight and input vector
//				float[] currLateralVar = calculateDiff(mLateralWeight, currentlateralInput, mLateralMasks);
//				updateWeights(mLateralVariance, mLateralMasks, currLateralVar, getAmnesicLearningRate(ages), false);
//				//if the neuron reach the stable stage, start lateral synaptic maintenance
//				if (mFiringAge > SMAGE) {
//					lateralMeanVariance = mean(mLateralVariance, mLateralMasks);
//					if (lateralMeanVariance < 0.01f) {
//						lateralMeanVariance = 0.01f;
//					}
//					//update the lateral mask
//					updateLateralMask(lateralMeanVariance);
//				}
			}
			if (mFiringAge > SMAGE) {
				mFiringAge = 0;
			}
		}
	}

//	public Allocation getBottomUpWeightsAllocation() {
//		return null;
//	}
//
//	public Allocation getLateralWeightsAllocation(){
//		return null;
//	}
//
//	public void copyBottomUpWeightsAllocation(Allocation weights) {
//	}
//
//	public Allocation getBottomUpMaskAllocation() {
//		return null;
//	}
//
//	public Allocation getTopDownWeightsAllocation() {
//		return null;
//	}
//
//	public Allocation getTopDownMaskAllocation() {
//		return null;
//	}
//
//
//	public void hebbianLearnHidden(Allocation sensorInput, Allocation motorInput, float[] lateralInput) {
//
//	}
//
//	public void destroyAllocations() {
//	}
//
//	public void saveAllocations() {
//	}
//
//	public Allocation getBottomUpAgeAllocation() {
//		return null;
//	}
//
//	public Allocation getTopDownAgeAllocation() {
//		return null;
//	}

	//		public float getMaxVolume() {
//			return maxVolume;
//		}
//
//		public void setMaxVolume(float newVolume) {
//			maxVolume = newVolume;
//		}
	public void setSerotoninMultiplier(float multiplier){
		serotoninMultiplier = multiplier;
	}

	public void setDopamineMultiplier(float multiplier){
		dopamineMultiplier = multiplier;
	}
}

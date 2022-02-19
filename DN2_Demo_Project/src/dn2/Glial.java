package dn2;
import java.io.Serializable;

/**
 * <p>
 * 		This class provides support to the Neurons and InhibitoryNeurons within the HiddenLayer and PrimaryHiddenLayer classes.
 * 		All the neuron class instances created by the DN2 are "born" within a 3D space, where each neuron can grow and die during the program's execution.
 *		Like its biological equivalent, these Glial instances help support each Neuron and InhibitoryNeuron as follows:
 *
 * 		<ul>
 * 			<li> Maintain the neurons in place within the 3D Space </li>
 * 			<li> Pull neurons into groups or clusters that share similar firing patterns </li>
 * 			<li> Remove "dead" neurons from the network </li>
 * 		</ul>
 *
 *
 * </p>
 *
 */
class Glial implements Serializable{


	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	/** Glial cell's 3D location */
	private float[] mLocation;

	/** Number of adjacent neurons each glial cell pulls towards itself */
	private int mTopK;

	/** Glial cell's current index */
	private int mIndex;

	/** Array of adjacent Glial cell indexes which are needed to be pulled */
	private int[] mPullIndex;

	//the vector which contains pulling vector for each neighbor neuron
	private float[][] mPullvectors;

	/**
	 * Glial cell constructor
	 *
	 * <p>
	 * 		This method initializes a single Glial cell instance.
	 * </p>
	 * @param topk Number of adjacent neurons this Glial cell will pull
	 * @param index This glial cell index
	 */
	public Glial(int topk, int index){

		//set the number of neighbor neurons each glial cell pulls
		mTopK = topk;
		//set the index
		mIndex = index;
		//construct the location vector
		// Location = (i,j,k)
		mLocation = new float[3];
		//construct the vector which recording neighbor neurons' index
		mPullIndex = new int[mTopK];
		//construct the vector which contains pulling vector for each neighbor neuron
		mPullvectors = new float[mTopK][3];
	}

	//calculate the distance between the two 3-D locations
	/**
	 * This method computes the Euclidean distance between two 3D locations
	 * @param vector 3D location of another Glial cell
	 * @return float Euclidean distance value
	 */
	public float computeDistance(float[] vector){
		//make sure the two location vector have same length
		// Only compute the distance if both location vectors have these length
		assert vector.length == mLocation.length;

		//calculate the distance
		float delta_h = (mLocation[0]-vector[0])*(mLocation[0]-vector[0]);
		float delta_v = (mLocation[1]-vector[1])*(mLocation[1]-vector[1]);
		float delta_d = (mLocation[2]-vector[2])*(mLocation[2]-vector[2]);
		float dis = (float)Math.sqrt((double)(delta_h+delta_v+delta_d));
		return dis;

	}

	//set the number of neighbor neurons each glial cell pulls
	/**
	 * This method assigns the number of k adjacent neurons each glial cell pulls.
	 * @param topk Number of adjacent neurons
	 */
	public void settopk(int topk){
		this.mTopK=topk;
	}

	//get the number of neighbor neurons each glial cell pulls
	/**
	 * Returns the number of neurons this Glial cell pulls
	 *
	 * @return int Number of adjacent neurons to pull
	 */
	public int gettopk(){
		return mTopK;
	}

	//set the index
	/**
	 *
	 * @param index
	 */
	public void setindex(int index){
		this.mIndex=index;
	}

	//get the index
	/**
	 *
	 * @return
	 */
	public int getindex(){
		return mIndex;
	}

	//set one neighbor neurons' index in corresponding location
	/**
	 *
	 * @param index
	 * @param input
	 */
	public void setpullindex(int index, int input){
		this.mPullIndex[index] = input;
	}

	//get the vector which recording neighbor neurons' index
	/**
	 *
	 * @param index
	 * @return
	 */
	public int getpullindex(int index){
		return mPullIndex[index];
	}

	//set one pulling vector for the neighbor neuron
	/**
	 *
	 * @param vector
	 * @param index
	 */
	public void setpullvector(float[] vector, int index){
		if(index < mTopK){
			this.mPullvectors[index] = vector;
		}
	}

	//get one pulling vector for the neighbor neuron
	/**
	 *
	 * @param index
	 * @return
	 */
	public float[] getpullvector(int index){
		return mPullvectors[index];
	}

	//set glial cell's location
	/**
	 *
	 * @param location
	 */
	public void setlocation(float[] location){
		this.mLocation = location;
	}

	//set glial cell's location
	/**
	 *
	 * @return
	 */
	public float[] getlocation(){
		return mLocation;
	}

	//set the vector which contains pulling vector for each neighbor neuron
	/**
	 *
	 * @param vectors
	 */
	public void setlpullvectors(float[][] vectors){
		this.mPullvectors = vectors;
	}

	//get the vector which contains pulling vector for each neighbor neuron
	/**
	 *
	 * @return
	 */
	public float[][] getpullvectors(){
		return mPullvectors;
	}


}
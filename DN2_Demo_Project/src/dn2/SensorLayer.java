package dn2;

import java.io.Serializable;

/**
 * <h1>SensorLayer</h1>
 *
 * <p>
 * This class receives user input and prepares it for the DN2.
 *
 * The inputs are modality-independent, which means that it does not matter how the user
 * sends the data, the DN receives it as a vector.
 *
 * Sample types of inputs can be images, character symbols, speech/audio recordings.
 *
 *
 * </p>
 */
class SensorLayer implements Serializable{

	/**
	 * Class serial ID.
	 *
	 * Identifies the array of bytes of the object sent across network sockets or text files.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The height dimension of the input of a single sensor.
	 */
	private int height;

	/**
	 * The width dimension of the input of a single sensor.
	 */
	private int width;

	/**
	 * 2D input of the sensor.
	 */
	private float[][] input;

	/**
	 * SensorLayer constructor.
	 * Initializes the SensorLayer with the 2 dimensions values.
	 *
	 * @param height
	 * @param width
	 */
	public SensorLayer(int height, int width){
		this.setWidth(width);
		this.setHeight(height);

		input = new float[height][width];
	}

	/**
	 * SensorLayer constructor.
	 * Initializes the SensorLayer with the 2 dimensions values and an initial input.
	 *
	 * @param height
	 * @param width
	 * @param input
	 */
	public SensorLayer(int height, int width, float[][] input){
		this.setWidth(width);
		this.setHeight(height);

		this.setInput(input);
	}

	/**
	 * Getter Method for the sensor's height dimension.
	 *
	 * @return The height of the sensor input
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * Setter Method for the sensor's height dimension.
	 *
	 * @param height The new height value for the sensor input
	 */
	public void setHeight(int height) {
		this.height = height;
	}

	/**
	 * Getter Method for the sensor's width dimension.
	 *
	 * @return The width of the sensor input
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * Setter Method for the sensor's width dimension.
	 *
	 * @param width The width of the sensor input
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * This method converts the current 2D sensor input into a 1D array. 
	 *
	 * @return 1D array of the sensor input.
	 */
	public float[] getInput1D() {
		float[] inputArray = new float[height * width];

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int index = i*width + j;
				inputArray[index] = input[i][j];
			}

		}

		return inputArray;
	}

	/**
	 * Getter method for the 2D input.
	 *
	 * @return the 2D input matrix.
	 */
	public float[][] getInput() {

//		float[][] copy = new float[height][width];
//		
//		for(int i=0; i < height; i++)
//			System.arraycopy(input, i*width, copy[i],  0, width);
//		
//		return copy;

		return input;
	}

	/**
	 * Setter method for the 2D input.
	 *
	 * @param input The current 2D input. 
	 */
	public void setInput(float[][] input) {
		this.input = input;
	}



}

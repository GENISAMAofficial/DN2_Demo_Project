package dn2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int numberOfNeurons = 100; //maximum number of neurons
        DNHandler handler = new DNHandler(numberOfNeurons, "001.txt", "002.csv", "neuronGrowth.txt"); //initialize DNHandler and pass growth tables
        handler.setFrozenDN(true); //do not learn while testing
        handler.setZtopk(new int[]{1}); //one motor predicted per update
        handler.setMaxYtopk(1); //one neuron firing per update (increase this number and optimize)
        handler.setPre_screen(.5f);
        handler.setNeuronTypes(new int[]{0,0,0,0,1,0,0}); //use type 5 neurons (bottom up and top down input only)
        handler.setComputeResponseWeights(new int[]{9,0,1}); //bottom up .9 and top down .1
        handler.setMotorComputeResponseWeights(new int[]{1,0});
        handler.setRfSizes(new int[] {3}, new int[] {3}); //initialize rf size (all of input currently)
        handler.setRfStrides(new int[] {3}, new int[] {3}); //initialize rf stride
        handler.setDopSerWeights(new float[] {.1f, .1f}); //set reinforcer weights
        handler.createDefaultGrowthRateTable(); //create default table
        
        //set the response types for hidden layer and motor layer (default is always goodness of match)
        handler.setHiddenResponseType(DNHandler.RESPONSE_CALCULATION_TYPE.GOODNESS_OF_MATCH);
        int numberOfMotors = 1;
        DNHandler.RESPONSE_CALCULATION_TYPE[] responseType = new DNHandler.RESPONSE_CALCULATION_TYPE[numberOfMotors];
        responseType[0] = DNHandler.RESPONSE_CALCULATION_TYPE.GOODNESS_OF_MATCH;
        handler.setMotorResponseType(responseType);
                
        //sensor and motor variables
        ArrayList<float[][]> sensor;
        ArrayList<float[][]> motor;
        ArrayList<float[][]> newMotor;
        
        //set training data
        sensor = new ArrayList<float[][]>();
        sensor.add(new float[][] {{1f, 0f, 0f},{0f, 0f, 0f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 1f, 0f},{0f, 0f, 0f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 1f},{0f, 0f, 0f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{1f, 0f, 0f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 1f, 0f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 1f},{0f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 0f},{1f, 0f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 0f},{0f, 1f, 0f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 0f},{0f, 0f, 1f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 0f},{0f, 0f, 1f}});
        sensor.add(new float[][] {{0f, 0f, 0f},{0f, 0f, 0f},{0f, 0f, 1f}});

        motor = new ArrayList<float[][]>();
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f}});
        motor.add(new float[][] {{1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f}});
        motor.add(new float[][] {{0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f}});

        //training
        newMotor = new ArrayList<float[][]>();
        for (int i = 0; i < sensor.size(); i++) {
        	ArrayList<float[][]> tempSensor = new ArrayList<float[][]>();
        	tempSensor.add(sensor.get(i));
        	
        	ArrayList<float[][]> tempMotor = new ArrayList<float[][]>();
        	tempMotor.add(motor.get(i));
        	
            newMotor = handler.Update(tempSensor, tempMotor, new boolean[] {false}, new boolean[] {false}, true);
        }
        
        //save DN to downloads folder
//TODO you must change this file path to your own file path where you want to save your DN!
        handler.saveDN("file_path"); // For Example: Users/Steve/GENISAMA/TempDN //Note: TempDN is the saved file name
        
        //attempt to load DN
        try {
			handler.loadDN("file_path"); //TODO change this path to match the above changed path
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			System.out.println("ClassNotFoundException" + e);
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("IOException" + e);
			e.printStackTrace();
		}
        
        //complete disjoint testing
        newMotor = new ArrayList<float[][]>();
        for (int i = 0; i < sensor.size(); i++) {
        	ArrayList<float[][]> tempSensor = new ArrayList<float[][]>();
        	tempSensor.add(sensor.get(i));
        	
        	ArrayList<float[][]> tempMotor = new ArrayList<float[][]>();
        	tempMotor.add(motor.get(i));
        	
            newMotor = handler.Update(tempSensor, tempMotor, new boolean[] {false}, new boolean[] {false}, true);
            
            //print response
            System.out.println(Arrays.toString(newMotor.get(0)[0]));
            
        }
        
		
	}

}

package dn2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class TableReader {
	private String mFilename;
	private int type;
	private float[][] vector;
//	private Context mContext;
    public TableReader(String filename){
//    	mContext = context;
    	mFilename = filename;
    	String[] temp = mFilename.split("\\.");
    	if(temp[1].equals("xls")){
    		type = 1;
    	}
    	else if(temp[1].equals("txt")||temp[1].equals("csv")){
    		type = 2;
    	}
    	else{
    		type = 0;
    	}
    }
    
    public float[][] getTable() throws IOException {
    	if(type == 2){
    		vector = readTxt();
    	}
    	return vector;
    }
    
    public float[][] readTxt() throws IOException {
    	//Scanner reader = new Scanner(new File(mFilename));
//		Scanner reader = new Scanner(mContext.getAssets().open(mFilename));
		Scanner reader = new Scanner(getClass().getClassLoader().getResourceAsStream(mFilename));
    	ArrayList<String> templist = new ArrayList<String>(); 
    	while(reader.hasNextLine()){
    		String temp = reader.nextLine();
    		if(temp.trim().matches("^[0-9].*")){
    			templist.add(temp);
    		}
    	}
    	reader.close();
    	float[][] list = new float[templist.size()][];
    	for(int i = 0; i < templist.size(); i++){
    		String[] temp2 = templist.get(i).split(","); 
    		list[i] = new float[temp2.length];
    		for(int j = 0; j < temp2.length; j++){
    			list[i][j] = Float.parseFloat(temp2[j].trim());
    		}
    	}
    	return list;
    }

}

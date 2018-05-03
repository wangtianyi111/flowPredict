package com.isaac.stock.representation;

import com.google.common.collect.ImmutableMap;
import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by zhanghao on 26/7/17. Modified by zhanghao on 28/9/17.
 * 
 * @author ZHANG HAO
 */
public class StockDataSetIterator implements DataSetIterator {

	/** category and its index */
	private final Map<PriceCategory, Integer> featureMapIndex = ImmutableMap.of(PriceCategory.WEEK, 0,
			PriceCategory.PERIOD, 1, PriceCategory.IN, 2,PriceCategory.OUT, 3);

	private final int VECTOR_SIZE = 4; // number of features for a stock data
	private int miniBatchSize; // mini-batch size
	private int exampleLength; // default 22, say, 22 working days per month
	private int predictLength = 1; // default 1, say, one day ahead prediction

	/** minimal values of each feature in stock dataset */
	private double[] minArray = new double[VECTOR_SIZE];
	/** maximal values of each feature in stock dataset */
	private double[] maxArray = new double[VECTOR_SIZE];

	/** feature to be selected as a training target */
	private PriceCategory category;

	/** mini-batch offset */
	private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

	/** stock dataset for training */
	private List<StationData> train;
	/** adjusted stock dataset for testing */
	private List<Pair<INDArray, INDArray>> test;

	public StockDataSetIterator(String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio,
			PriceCategory category) {
		List<StationData> stationDataList = readStationDataFromFile(filename, symbol);
		System.out.println("stationDataList.size()="+stationDataList.size());
		this.miniBatchSize = miniBatchSize;
		this.exampleLength = exampleLength;
		this.category = category;
		// 划分训练集和测试集
		int split = (int) Math.round(stationDataList.size() * splitRatio);
		train = stationDataList.subList(0, split);
		test = generateTestDataSet(stationDataList.subList(split, stationDataList.size()));
		initializeOffsets();
	}

	/** initialize the mini-batch offsets */
	private void initializeOffsets() {
		exampleStartOffsets.clear();
		int window = exampleLength + predictLength;
		for (int i = 0; i < train.size() - window; i++) {
			exampleStartOffsets.add(i);
		}
	}

	public List<Pair<INDArray, INDArray>> getTestDataSet() {
		return test;
	}

	public double[] getMaxArray() {
		return maxArray;
	}

	public double[] getMinArray() {
		return minArray;
	}

	public double getMaxNum(PriceCategory category) {
		return maxArray[featureMapIndex.get(category)];
	}

	public double getMinNum(PriceCategory category) {
		return minArray[featureMapIndex.get(category)];
	}

	@Override
	public DataSet next(int num) {
		if (exampleStartOffsets.size() == 0)
			throw new NoSuchElementException();
		int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
		INDArray input = Nd4j.create(new int[] { actualMiniBatchSize, VECTOR_SIZE, exampleLength }, 'f');
		INDArray label;
		if (category.equals(PriceCategory.ALL))
			label = Nd4j.create(new int[] { actualMiniBatchSize, VECTOR_SIZE, exampleLength }, 'f');
		else
			label = Nd4j.create(new int[] { actualMiniBatchSize, predictLength, exampleLength }, 'f');
		for (int index = 0; index < actualMiniBatchSize; index++) {
			int startIdx = exampleStartOffsets.removeFirst();
			int endIdx = startIdx + exampleLength;
			StationData curData = train.get(startIdx);
			StationData nextData;
			for (int i = startIdx; i < endIdx; i++) {
				int c = i - startIdx;
				//这里需要修改
				input.putScalar(new int[] { index, 0, c },
						(curData.getWeek() - minArray[0]) / (maxArray[0] - minArray[0]));
				input.putScalar(new int[] { index, 1, c },
						(curData.getPeriod() - minArray[1]) / (maxArray[1] - minArray[1]));
				input.putScalar(new int[] { index, 2, c },
						(curData.getIn() - minArray[2]) / (maxArray[2] - minArray[2]));
				input.putScalar(new int[] { index, 3, c },
						(curData.getOut() - minArray[3]) / (maxArray[3] - minArray[3]));
				nextData = train.get(i + 1);
				if (category.equals(PriceCategory.ALL)) {
					label.putScalar(new int[] { index, 0, c },
							(nextData.getWeek() - minArray[0]) / (maxArray[0] - minArray[0]));
					label.putScalar(new int[] { index, 1, c },
							(nextData.getPeriod() - minArray[1]) / (maxArray[1] - minArray[1]));
					label.putScalar(new int[] { index, 2, c },
							(nextData.getIn() - minArray[2]) / (maxArray[2] - minArray[2]));
					label.putScalar(new int[] { index, 3, c },
							(nextData.getOut() - minArray[3]) / (maxArray[3] - minArray[3]));
				} else {
					label.putScalar(new int[] { index, 0, c }, feedLabel(nextData));
				}
				curData = nextData;
			}
			if (exampleStartOffsets.size() == 0)
				break;
		}
		return new DataSet(input, label);
	}
	//这里需要修改
	private double feedLabel(StationData data) {
		double value;
		switch (category) {
		case WEEK:
			value = (data.getWeek() - minArray[0]) / (maxArray[0] - minArray[0]);
			break;
		case PERIOD:
			value = (data.getWeek() - minArray[1]) / (maxArray[1] - minArray[1]);
			break;
		case IN:
			value = (data.getIn() - minArray[2]) / (maxArray[2] - minArray[2]);
			break;
		case OUT:
			value = (data.getOut() - minArray[3]) / (maxArray[3] - minArray[3]);
			break;
		default:
			throw new NoSuchElementException();
		}
		return value;
	}

	@Override
	public int totalExamples() {
		return train.size() - exampleLength - predictLength;
	}

	@Override
	public int inputColumns() {
		return VECTOR_SIZE;
	}

	@Override
	public int totalOutcomes() {
		if (this.category.equals(PriceCategory.ALL))
			return VECTOR_SIZE;
		else
			return predictLength;
	}

	@Override
	public boolean resetSupported() {
		return false;
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public void reset() {
		initializeOffsets();
	}

	@Override
	public int batch() {
		return miniBatchSize;
	}

	@Override
	public int cursor() {
		return totalExamples() - exampleStartOffsets.size();
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
		throw new UnsupportedOperationException("Not Implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not Implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not Implemented");
	}

	@Override
	public boolean hasNext() {
		return exampleStartOffsets.size() > 0;
	}

	@Override
	public DataSet next() {
		return next(miniBatchSize);
	}
	//这里需要修改
	private List<Pair<INDArray, INDArray>> generateTestDataSet(List<StationData> stationDataList) {
		int window = exampleLength + predictLength;
		List<Pair<INDArray, INDArray>> test = new ArrayList<>();
		for (int i = 0; i < stationDataList.size() - window; i++) {
			INDArray input = Nd4j.create(new int[] { exampleLength, VECTOR_SIZE }, 'f');
			for (int j = i; j < i + exampleLength; j++) {
				StationData stock = stationDataList.get(j);
				input.putScalar(new int[] { j - i, 0 }, (stock.getWeek() - minArray[0]) / (maxArray[0] - minArray[0]));
				input.putScalar(new int[] { j - i, 1 }, (stock.getPeriod() - minArray[1]) / (maxArray[1] - minArray[1]));
				input.putScalar(new int[] { j - i, 2 }, (stock.getIn() - minArray[2]) / (maxArray[2] - minArray[2]));
				input.putScalar(new int[] { j - i, 3 }, (stock.getOut() - minArray[3]) / (maxArray[3] - minArray[3]));
			}
			StationData stock = stationDataList.get(i + exampleLength);
			INDArray label;
			if (category.equals(PriceCategory.ALL)) {
				label = Nd4j.create(new int[] { VECTOR_SIZE }, 'f'); // ordering is set as 'f', faster construct
				label.putScalar(new int[] { 0 }, stock.getWeek());
				label.putScalar(new int[] { 1 }, stock.getPeriod());
				label.putScalar(new int[] { 2 }, stock.getIn());
				label.putScalar(new int[] { 3 }, stock.getOut());
			} else {
				label = Nd4j.create(new int[] { 1 }, 'f');
				switch (category) {
				case WEEK:
					label.putScalar(new int[] { 0 }, stock.getWeek());
					break;
				case PERIOD:
					label.putScalar(new int[] { 0 }, stock.getPeriod());
					break;
				case IN:
					label.putScalar(new int[] { 0 }, stock.getIn());
					break;
				case OUT:
					label.putScalar(new int[] { 0 }, stock.getOut());
					break;
				default:
					throw new NoSuchElementException();
				}
			}
			test.add(new Pair<>(input, label));
		}
		return test;
	}

	private List<StationData> readStationDataFromFile(String filename, String symbol) {
		List<StationData> stationDataList = new ArrayList<>();
		try {
			for (int i = 0; i < maxArray.length; i++) { // initialize max and min arrays
				maxArray[i] = Double.MIN_VALUE;
				minArray[i] = Double.MAX_VALUE;
			}
			List<String[]> list = new CSVReader(new FileReader(filename)).readAll(); // load all elements in a list
			for (String[] arr : list) {
				if (!arr[1].equals(symbol))
					continue;
				double[] nums = new double[VECTOR_SIZE];
				for (int i = 0; i < arr.length - 2; i++) {
					nums[i] = Double.valueOf(arr[i + 2]);
					if (nums[i] > maxArray[i])
						maxArray[i] = nums[i];
					if (nums[i] < minArray[i])
						minArray[i] = nums[i];
				}
				stationDataList.add(new StationData(arr[0], arr[1], nums[0], nums[1], nums[2], nums[3]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return stationDataList;
	}
}

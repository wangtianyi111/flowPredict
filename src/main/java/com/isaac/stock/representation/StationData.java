package com.isaac.stock.representation;

public class StationData {
	private String day; // 几号
	private String symbol; // 站点名

	private double week; // 周几
	private double period;//时间段
	private double in; // 进站客流
	private double out; // 出战客流

	public StationData() {
		
	}
	public StationData(String day, String symbol, double week,double period, double in, double out) {
		this.day = day;
		this.symbol = symbol;
		this.week = week;
		this.period = period;
		this.in = in;
		this.out = out;
	}
	
	public double getPeriod() {
		return period;
	}
	public void setPeriod(double period) {
		this.period = period;
	}
	public String getDay() {
		return day;
	}
	public void setDay(String day) {
		this.day = day;
	}
	public String getSymbol() {
		return symbol;
	}
	public void setSymbol(String symbol) {
		this.symbol = symbol;
	}
	public double getWeek() {
		return week;
	}
	public void setWeek(double week) {
		this.week = week;
	}
	public double getIn() {
		return in;
	}
	public void setIn(double in) {
		this.in = in;
	}
	public double getOut() {
		return out;
	}
	public void setOut(double out) {
		this.out = out;
	}
	
}

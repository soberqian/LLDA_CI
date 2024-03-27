package com.qian.runexample;

import com.qian.model.LinkLDACVB;

public class Test {

	public static void main(String[] args) {
//		LinkLDACVB linklda = new LinkLDACVB("test/data/programmableweb.txt", "gbk", "\t", ",", 10, 0.1,
//				0.01,0.01, 100, 80, "test/output/");
//		linklda.CVBInference();
		LinkLDACVB linklda = new LinkLDACVB("test/data/cardatatest.txt", "utf-8", "====", "_", 15, 0.1,
				0.01,0.01, 300, 50, "test/output/");
		linklda.CVBInference();
	}

}

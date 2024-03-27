package com.qian.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.qian.util.FileUtil;
import com.qian.util.FuncUtils;



/**
 * 
 * collapsed variational Bayesian (CVB) inference for Link LDA
 * @author: Yang Qian (HeFei University of Technology)
 */


public class LinkLDACVB
{
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public double gamma; // Hyper-parameter gamma
	public int K; // number of topics
	public int iterations; // number of iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int [][] docword;//word index array
	public int L; // number of links in the corpus
	public int [][] doclink;//link index array
	//for link LDA
	public Map<String, Integer> linkToIndexMap = new HashMap<String, Integer>(); //link to index
	public List<String> indexLinkMap = new ArrayList<String>();   //index to String link 
	//word related variational parameters
	public double[][] ndk; // document-topic count
	public double[] ndsum; //document-topic sum
	public double[][] nkw; //topic-word count
	public double[] nksum_w; //topic-word sum (total number of words assigned to a topic)
	public double[][][] gamma_word; 
	//link related variational parameters
	public double[][] nkl;  //topic-link count
	public double[] nksum_l; //topic-link sum 
	public double[][][] gamma_link; 
	//output
	public int topWordsAndLinksOutputNumber;
	public String outputFileDirectory; 
	public String outputFilecode; 
	public LinkLDACVB(String inputFile, String inputFileCode, String separator_wordAndLink, String separator_link, int topicNumber,
			double inputAlpha, double inputBeta,double inputGamma, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		doclink = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			List<String> links = new ArrayList<String>();
			FileUtil.tokenize(line.split(separator_wordAndLink)[1], words);
			//					FileUtil.tokenizeAndLowerCase(line.split("====")[1], words);
			FileUtil.tokenizeEntity(line.split(separator_wordAndLink)[0], links,separator_link);
			docword[j] = new int[words.size()];
			for(int i = 0; i < words.size(); i++){
				String word = words.get(i);
				if(!wordToIndexMap.containsKey(word)){
					int newIndex = wordToIndexMap.size();
					wordToIndexMap.put(word, newIndex);
					indexToWordMap.add(word);
					docword[j][i] = newIndex;
				} else {
					docword[j][i] = wordToIndexMap.get(word);
				}
			}
			doclink[j] = new int[links.size()];
			for(int i = 0; i < links.size(); i++){
				String link = links.get(i);
				if(!linkToIndexMap.containsKey(link)){
					int newIndex = linkToIndexMap.size();
					linkToIndexMap.put(link, newIndex);
					indexLinkMap.add(link);
					doclink[j][i] = newIndex;
				} else {
					doclink[j][i] = linkToIndexMap.get(link);
				}
			}
			j++;

		}
		V = indexToWordMap.size();
		L = indexLinkMap.size();
		System.out.println(V  + "\t" + L);
		alpha = inputAlpha;
		beta = inputBeta;
		gamma = inputGamma;
		K = topicNumber;
		iterations = inputIterations;
		topWordsAndLinksOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		outputFilecode = inputFileCode;
		initialize();
	}
	/**
	 * Randomly initialize the variational parameter using Gaussian
	 */
	public void initialize(){
		int D = docword.length;
		//variational parameters
		ndk = new double[D][K];
		ndsum = new double[D];
		nkw = new double[K][V];
		nksum_w = new double[K];
		gamma_word = new double[D][][]; 
		nkl = new double[K][L];
		nksum_l = new double[K];
		gamma_link = new double[D][][]; 
		for (int d = 0; d < D; d++) {
			int Nd = docword[d].length;
			gamma_word[d] = new double[Nd][K];
			for(int n = 0; n < Nd; n ++) {
				gamma_word[d][n] = FuncUtils.getGaussianSample(K, 0.5, 0.5);
				double gamma_norm = 0;
				for(int k = 0; k < K; k ++) {
					gamma_norm += Math.exp(gamma_word[d][n][k]);
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] = Math.exp(gamma_word[d][n][k]) / gamma_norm;
					nksum_w[k] += gamma_word[d][n][k];
					ndk[d][k] += gamma_word[d][n][k];
					nkw[k][docword[d][n]] += gamma_word[d][n][k];
					ndsum[d] += gamma_word[d][n][k];
				}
			}
			int NLink = doclink[d].length;
			gamma_link[d] = new double[NLink][K];
			for(int n = 0; n < NLink; n ++) {
				gamma_link[d][n] = FuncUtils.getGaussianSample(K, 0.5, 0.5);
				double gamma_norm = 0;
				for(int k = 0; k < K; k ++) {
					gamma_norm += Math.exp(gamma_link[d][n][k]);
				}
				for(int k = 0; k < K; k ++) {
					gamma_link[d][n][k] = Math.exp(gamma_link[d][n][k]) / gamma_norm;
					nksum_l[k] += gamma_link[d][n][k];
					nkl[k][doclink[d][n]] += gamma_link[d][n][k];
					ndk[d][k] += gamma_link[d][n][k];
					ndsum[d] += gamma_link[d][n][k];
				}
			}
		}

	}
	public void CVBInference(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			iterateCVB0Update();
		}
		// output the result
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		System.out.println("write topic link ..." );
		writeTopLinks();
		writeTopLinksWithProbability();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		writeTopWords();

	}
	public void iterateCVB0Update() {
		int D = docword.length;
		for(int d = 0; d < D; d ++) {
			for(int n = 0; n < docword[d].length; n ++) {
				double norm_w = 0;
				double[] gamma_w = new double[K];
				for(int k = 0; k < K; k ++) {
					gamma_w[k] = gamma_word[d][n][k];
					gamma_word[d][n][k] = (updateCount(d, n, k, 0, d) + alpha)*
							(beta + updateCount(d, n, k, docword[d][n], -1))/(V * beta + updateCount(d, n, k, 0, -1));
					norm_w += gamma_word[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] /= norm_w;
					//update
					nksum_w[k] += gamma_word[d][n][k] - gamma_w[k];
					ndk[d][k] += gamma_word[d][n][k] - gamma_w[k];
					nkw[k][docword[d][n]] += gamma_word[d][n][k] - gamma_w[k];
					ndsum[d] += gamma_word[d][n][k] - gamma_w[k];
				}
			}
			for(int n = 0; n < doclink[d].length; n ++) {
				double norm_l = 0;
				double[] gamma_l = new double[K];
				for(int k = 0; k < K; k ++) {
					gamma_l[k] = gamma_link[d][n][k];
					gamma_link[d][n][k] = (updateCount_Link(d, n, k, 0, d) + alpha)*
							(gamma + updateCount_Link(d, n, k, doclink[d][n], -1))/(L * gamma + updateCount_Link(d, n, k, 0, -1));
					norm_l += gamma_link[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_link[d][n][k] /= norm_l;
					//update
					nksum_l[k] += gamma_link[d][n][k] - gamma_l[k];
					ndk[d][k] += gamma_link[d][n][k] - gamma_l[k];
					nkl[k][doclink[d][n]] += gamma_link[d][n][k] - gamma_l[k];
					ndsum[d] += gamma_link[d][n][k] - gamma_l[k];
				}
			}
		}
	}
	/**
	 * update the count 
	 * expect the word d_n
	 * @param 
	 * @return
	 */
	public double updateCount(int d, int n, int k, int wsdn, int doc) {
		if(wsdn == 0 && doc == -1)
			return nksum_w[k] - gamma_word[d][n][k];
		else if(doc == -1)
			return nkw[k][wsdn] - gamma_word[d][n][k];
		else
			return ndk[doc][k] - gamma_word[d][n][k];
	}
	/**
	 * update the count for link information
	 * @param 
	 * @return
	 */
	public double updateCount_Link(int d, int n, int k, int link, int doc) {
		if(link == 0 && doc == -1)
			return nksum_l[k] - gamma_link[d][n][k];
		else if(doc == -1)
			return nkl[k][link] - gamma_link[d][n][k];
		else
			return ndk[doc][k] - gamma_link[d][n][k];
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha);
			}
		}
		return theta;
	}
	/**
	 * obtain the parameter Phi
	 */
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum_w[k] + V * beta);
			}
		}
		return phi;
	}
	/**
	 * obtain the parameter Sigma
	 */
	public double[][] estimateSigma() {
		double[][] sigma = new double[K][L];
		for (int k = 0; k < K; k++) {
			for (int l = 0; l < L; l++) {
				sigma[k][l] = (nkl[k][l] + gamma) / (nksum_l[k] + L * gamma);
			}
		}
		return sigma;
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWordsWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "topic_word_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWords(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		for (double[] phi_z : phi) {
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + "\t");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "topic_wordnop_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top links with probability for each topic
	 */
	public void writeTopLinksWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] sigma = estimateSigma();
		int topicNumber = 1;
		for (double[] sigma_z : sigma) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(sigma_z);
				sBuilder.append(indexLinkMap.get(max_index) + " :" + sigma_z[max_index] + "\n");
				sigma_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "topic_link_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top links with probability for each topic
	 */
	public void writeTopLinks(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] sigma = estimateSigma();
		for (double[] sigma_z : sigma) {
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(sigma_z);
				sBuilder.append(indexLinkMap.get(max_index) + "\t");
				sigma_z[max_index] = 0;
			}
			sBuilder.append("\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "topic_linknop_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writeDocumentTopic(){
		double[][] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int d = 0; d < theta.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < theta[d].length; k++) {
				doc.append(theta[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "doc_topic_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{

	}
}

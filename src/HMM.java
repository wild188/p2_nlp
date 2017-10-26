import java.util.Set;
import java.util.Hashtable;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Objects;

import Jama.Matrix;

class IntPair{
	public Integer one;
	public Integer two;
	public IntPair(int x, int y){
		one = x;
		two = y;
	}

	@Override
	public boolean equals(Object x) {
		IntPair b = (IntPair)x;
		if(b == null) return false;
		return this.one.equals(b.one) && this.two.equals(b.two);
	}

	@Override
	public int hashCode() {
		return Objects.hash(one, two);
	}
}

class HMM {
	/* Section for variables regarding the data */
	
	//
	private ArrayList<Sentence> labeled_corpus;
	
	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;
	
	//Tracks number of occurances of each POS tag
	ArrayList<Integer> posCount;

	//count of paired POSs
	Hashtable<IntPair, Integer> posBigrams;

	//count of word mapping to a POS tag (word first POS second)
	Hashtable<IntPair, Integer> wordPOSBigram;

	// mapping POS tags in String to their indices
	Hashtable<String, Integer> pos_tags;
	
	// inverse of pos_tags: mapping POS tag indices to their String format
	Hashtable<Integer, String> inv_pos_tags;
	
	// vocabulary size
	int num_words;

	Hashtable<String, Integer> vocabulary;

	private int max_sentence_length;
	
	/* Section for variables in HMM */
	
	// transition matrix
	private Matrix A;
	private Matrix oA;

	// emission matrix
	private Matrix B;
	private Matrix oB;

	// prior of pos tags
	private Matrix pi;
	private Matrix opi;

	// store the scaled alpha and beta
	private Matrix alpha;
	
	private Matrix beta;

	// scales to prevent alpha and beta from underflowing
	private Matrix scales;

	// logged v for Viterbi
	//private Matrix v;
	private double[] v;
	//private Matrix back_pointer;
	private int[][] back_pointer;
	//private Matrix pred_seq;
	private int[] pred_seq;

	// \xi_t(i): expected frequency of pos tag i at position t. Use as an accumulator.
	private Matrix gamma;
	
	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at position t.  Use as an accumulator.
	private Matrix digamma;
	
	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private Matrix gamma_w;

	// \xi_0(i): expected frequency of pos tag i at position 0.
	private Matrix gamma_0;
	
	/* Section of parameters for running the algorithms */

	// smoothing epsilon for the B matrix (since there are likely to be unseen words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.01;

	// number of iterations of EM
	private int max_iters = 30;
	
	// \mu: a value in [0,1] to balance estimations from MLE and EM
	// \mu=1: totally supervised and \mu = 0: use MLE to start but then use EM totally.
	private double mu = 0.8;
	

	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
		this.labeled_corpus = _labeled_corpus;
		this.unlabeled_corpus = _unlabeled_corpus;

		//prepareMatrices();
	}

	private void print(Matrix out){
		for(int i = 0; i < out.getRowDimension(); i++){
			for(int j = 0; j < out.getColumnDimension(); j++){
				System.out.printf("%f ", out.get(i, j));
			}
			System.out.println();
		}
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
		Integer posCounter = 0;
		Integer vocabCounter = 0;
		posCount = new ArrayList<Integer>();
		pos_tags = new Hashtable<String, Integer>();
		inv_pos_tags = new Hashtable<Integer, String>();
		vocabulary = new Hashtable<String, Integer>();
		wordPOSBigram = new Hashtable<IntPair, Integer>();
		posBigrams = new Hashtable<IntPair, Integer>();

		for(Sentence s : labeled_corpus){
			int len = s.length();
			if(len > max_sentence_length) max_sentence_length = len;
			String prevPosTag = null;
			for(int i = 0; i < len; i++){
				Word curWord = s.getWordAt(i);
				String curPosTag;
				if(!pos_tags.containsKey(curPosTag = curWord.getPosTag())){
					pos_tags.put(curPosTag, posCounter);
					inv_pos_tags.put(posCounter, curPosTag);
					posCount.add(1);
					posCounter++;
				}else{
					int index = pos_tags.get(curPosTag);
					posCount.set(index, posCount.get(index) + 1);
				}
				if(!vocabulary.containsKey(curWord.getLemme())){
					vocabulary.put(curWord.getLemme(), vocabCounter);
					vocabCounter++;
				}

				//increment bigrams
				IntPair wordTag = new IntPair(vocabulary.get(curWord.getLemme()), pos_tags.get(curPosTag));
				if(wordPOSBigram.containsKey(wordTag)){
					//System.out.println("Hello!");
					wordPOSBigram.put(wordTag, wordPOSBigram.get(wordTag) + 1);
					//System.out.println(wordPOSBigram.get(wordTag));
				}else{
					wordPOSBigram.put(wordTag, 1);
				}
				if(i != 0){
					IntPair pos_pos = new IntPair(pos_tags.get(prevPosTag), pos_tags.get(curPosTag));
					if(posBigrams.containsKey(pos_pos)){
						posBigrams.put(pos_pos, posBigrams.get(pos_pos) + 1);
					}else{
						posBigrams.put(pos_pos, 1);
						//System.out.printf("Inserted %d, %d to pos bigrams.\n", pos_tags.get(prevPosTag), pos_tags.get(curPosTag));
					}
				}
				prevPosTag = curPosTag;
			}
		}

		//Add vocab from testing set
		for(Sentence s : unlabeled_corpus){
			int len = s.length();
			if(len > max_sentence_length) max_sentence_length = len;
			for(int i = 0; i < len; i++){
				Word curWord = s.getWordAt(i);
				if(!vocabulary.containsKey(curWord.getLemme())){
					vocabulary.put(curWord.getLemme(), vocabCounter);
					vocabCounter++;
				}
			}
		}

		num_postags = posCounter;
		num_words = vocabCounter;

		A = new Matrix(num_postags, num_postags);
		B = new Matrix(num_postags, num_words);
		pi = new Matrix(1, num_postags);
		
		mle();
	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
		int wordCount = 0;
		for(int i = 0; i < A.getRowDimension(); i++){
			for(int j = 0; j < A.getColumnDimension(); j++){
				IntPair pair = new IntPair(i, j);
				if(posBigrams.containsKey(pair)){
					double aij = ((double)posBigrams.get(pair) + smoothing_eps) / ((double)posCount.get(pair.one) + (smoothing_eps * num_postags * num_postags));
					A.set(i, j, aij);
					//xp++;
				}else{
					double aij = (smoothing_eps) / ((double)posCount.get(pair.one) + (smoothing_eps * num_postags * num_postags));
					A.set(i, j, aij);
				}
			}
			normalize(i, A);
		}

		int oneMatch = 0;
		int xp = 0;
		for(int i = 0; i < B.getRowDimension(); i++){
			for(int j = 0; j < B.getColumnDimension(); j++){
				IntPair pair = new IntPair(j, i);
				if(wordPOSBigram.containsKey(pair)){
					double bij = ((double)wordPOSBigram.get(pair) + smoothing_eps) / ((double)posCount.get(pair.two) + (smoothing_eps * num_postags * num_postags));
					B.set(pair.two, pair.one, bij);
					xp++;
				}else{
					double bij = (smoothing_eps) / ((double)posCount.get(i) + (smoothing_eps * num_postags * num_postags));
					B.set(i, j, bij);
				}
			}
			normalize(i, B);
		}


		int[] startPOS = new int[num_postags];
		int numSentences = 0;
		for(Sentence s : labeled_corpus){
			String tag = s.getWordAt(0).getPosTag();
			int index = pos_tags.get(tag);
			startPOS[index]++;
			numSentences++;
		}
		for(int i = 0; i < num_postags; i++){
			double value = (double)startPOS[i]/(double)numSentences;
			//System.out.printf("%s : %f\n", inv_pos_tags.get(i), value);
			pi.set(0, i, value);
		}

		//System.out.printf("%d word pos tag matches\n", wordCount);
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
		//if(Double.compare(mu, 1.0) == 0) return;
		oA = (Matrix)A.clone();
		oB = (Matrix)B.clone();
		opi = (Matrix)pi.clone();


		gamma = new Matrix(max_sentence_length, num_postags);
		digamma = new Matrix(num_postags, num_postags);
		gamma_w = new Matrix(num_postags, num_words);
		gamma_0 = new Matrix(1, num_postags);

		for(int i = 0; i < max_iters; i++){
			for(Sentence s : unlabeled_corpus){
				if(i == 0){
					s.prob = new double[max_iters];
				}
				s.prob[i] = expectation(s);
			}
			maximization();
		}
	}

	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
		double[] vConfidence = new double[unlabeled_corpus.size()];
		int i = 0;
		for(Sentence s : unlabeled_corpus){
			vConfidence[i] = viterbi(s);
		}
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		
		for(Sentence s : unlabeled_corpus){
			for(Word word : s){
				bw.write(word.getLemme() + " " + word.getPosTag());
				bw.newLine();
			}
			bw.newLine();
		}
		bw.close();
        fw.close();
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
		FileWriter fw = new FileWriter(outFileName);
		BufferedWriter bw = new BufferedWriter(fw);
		
		for(Sentence s : unlabeled_corpus){
			for(double prob : s.prob){
				bw.write(prob + ", ");
			}
			//bw.write(s.prob + " : ");
			for(Word word : s){
				bw.write(word.getLemme() + " ");
			}
			bw.newLine();
		}
		bw.close();
        fw.close();
	}
	
	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expectation(Sentence s) {
		double out = forward(s);

		backward(s);
		for(int t = 0; t < s.length(); t++){
			Word w = s.getWordAt(t);
			int wordIndex = vocabulary.get(w.getLemme());

			//Increment digamma for transitional accumulator
			if(t == (s.length() - 1)){
				for(int i = 0; i < num_postags; i++){
					for(int j = 0; j < num_postags; j++){
						double dgij = alpha.get(t, i) * A.get(i, j) * B.get(j, wordIndex);
						double newVal = dgij + digamma.get(i, j);
						digamma.set(i, j, newVal);
					}
				}
			}else{
				for(int i = 0; i < num_postags; i++){
					for(int j = 0; j < num_postags; j++){
						double dgij = alpha.get(t, i) * A.get(i, j) * B.get(j, wordIndex) * beta.get(t + 1, j);
						double newVal = dgij + digamma.get(i, j);
						digamma.set(i, j, newVal);
					}
				}
			} 
			

			//Increment gamma for location of POS tags within the sentences
			//int offset = max_sentence_length - s.length();
			for(int i = 0; i < num_postags; i++){
				double gti = alpha.get(t, i) * beta.get(t, i);
				double newVal = gamma.get(t, i) + gti;
				gamma.set((t), i, newVal);

				if(t == 0){
					newVal = gamma_0.get(0, i) * gti;
					gamma_0.set(0, i, newVal);
				}

				//Increments probability that POS tag i emits word w
				gamma_w.set(i, wordIndex, (gamma_w.get(i, wordIndex) + gti));
			}
		}
		
		//print(beta);
		return out;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
		Matrix ahat = new Matrix(num_postags, num_postags);
		Matrix bhat = new Matrix(num_postags, num_words);
		

		for(int i = 0; i < num_postags; i++){
			double expectedI = 0;
			for(int k = 0; k < max_sentence_length - 1; k++){
				expectedI += gamma.get(k, i);
			}

			for(int j = 0; j < num_postags; j++){
				ahat.set(i, j, (digamma.get(i, j) / expectedI));
			}

			for(int h = 0; h < num_words; h++){
				bhat.set(i, h, (gamma_w.get(i, h) / (expectedI + gamma.get(max_sentence_length - 1, i))));
			}
		}

		// re-estimate
		for(int i = 0; i < num_postags; i++){
			for(int j = 0; j < num_postags; j++){
				double dVal = (1 - mu) * ahat.get(i, j);
				double lVal = mu * oA.get(i, j);
				A.set(i, j, lVal + dVal);
			}

			for(int j = 0; j < num_words; j++){
				double dVal = (1 - mu) * bhat.get(i, j);
				double lVal = mu * oB.get(i, j);
				B.set(i, j, lVal + dVal);
			}

			double dVal = (1 - mu) * gamma_0.get(0, i);
			double lVal = mu * opi.get(0, i);
			pi.set(0, i, lVal + dVal);
		}
	}

	private void normalize(int position, Matrix target){
		double sum = 0.0;
		for(int i = 0; i < num_postags; i++){
			sum+= target.get(position, i);
		}
		normalize(sum, position, target);
	}

	private void normalize(double sum, int position, Matrix target){
		double normalizeFactor = 1.0 / sum;
		for(int i = 0; i < num_postags; i++){
			target.set(position, i, (target.get(position, i) * normalizeFactor));
		}
	}

	private double forwardHelper(Sentence s, int position){
		Word word = s.getWordAt(position);
		int wIndex = vocabulary.get(word.getLemme());
		double result;
		double sum = 0;
		double max = 0;
		int maxPOSindex = 0;
		if(position == 0){
			for(int i = 0; i < num_postags; i++){
				double temp = pi.get(0, i) * B.get(i, wIndex);
				sum += temp;
				alpha.set(position, i, temp);
			}
			result = Math.log(sum);
		}else{
			double prevResult = forwardHelper(s, position - 1);
			Word prevWord = s.getWordAt(position - 1);
			//int prevPOSindex = pos_tags.get(prevWord.getPosTag());
			for(int j = 0; j < num_postags; j++){
				double temp = 0.0; //A.get(prevPOSindex, i) * B.get(i, wIndex);
				for(int i = 0; i < num_postags; i++){
					temp+= A.get(i, j) * B.get(j, wIndex) * alpha.get(position-1, i);
				}
				sum += temp;
				alpha.set(position, j, temp);
			}
			result = Math.log(sum) + prevResult;
		}
		normalize(sum, position, alpha);
		return result;
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
		alpha = new Matrix(s.length(), num_postags);
		return forwardHelper(s, s.length() - 1);
	}
 
	private double backwardHelper(Sentence s, int position){
		Word word = s.getWordAt(position);
		int wIndex = vocabulary.get(word.getLemme());
		double result;
		double sum = 0;
		double max = 0;
		int maxPOSindex = 0;
		if(position == (s.length() - 1)){
			for(int i = 0; i < num_postags; i++){
				double temp = B.get(i, wIndex);
				sum += temp;
				beta.set(position, i, temp);
			}
			result = Math.log(sum);
		}else{
			double prevResult = backwardHelper(s, position + 1);
			Word prevWord = s.getWordAt(position - 1);
			//int prevPOSindex = pos_tags.get(prevWord.getPosTag());
			for(int j = 0; j < num_postags; j++){
				double temp = 0;
				for(int i = 0; i < num_postags; i++){
					temp+= A.get(j, i) * B.get(j, wIndex) * beta.get(position + 1, i);
				}
				//double temp = A.get(prevPOSindex, j) * B.get(j, wIndex);
				sum += temp;
				beta.set(position, j, temp);
			}
			result = Math.log(sum) + prevResult;
		}
		//word.setPosTag(inv_pos_tags.get(new Integer(maxPOSindex)));
		normalize(sum, position, beta);
		return result;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
		beta = new Matrix(s.length(), num_postags);
		return backwardHelper(s, 0);
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 * private Matrix v;
	 * private Matrix back_pointer;
	 * private Matrix pred_seq;
	 */
	private double viterbi(Sentence s) {
		int len = s.length();
		back_pointer = new int[len][num_postags];
		v = new double[num_postags];//new Matrix(1, num_postags);
		Word word = s.getWordAt(0);
		int wIndex = vocabulary.containsKey(word.getLemme()) ? vocabulary.get(word.getLemme()) : -1;
		for(int k = 0; k < num_postags; k++){
			double prob = wIndex < 0 ? pi.get(0, k) * smoothing_eps : pi.get(0, k) * B.get(k, wIndex);
			v[k] = Math.log(prob);
			back_pointer[0][k] = k;
		}
		for(int i = 1; i < len; i++){
			word = s.getWordAt(i);
			wIndex = vocabulary.containsKey(word.getLemme()) ? vocabulary.get(word.getLemme()) : -1;
			for(int j = 0; j < num_postags; j++){
				double max = 0;
				int maxPos = 0;
				for(int h  = 0; h < num_postags; h++){
					double temp = wIndex < 0 ? A.get(j, h) * smoothing_eps : A.get(j, h) * B.get(h, wIndex);
					if(temp > max){
						max = temp;
						maxPos = h;
					}
				}
				back_pointer[i][j] = maxPos;
				v[j] += Math.log(max);
			}
		}
		
		int maxEndIndex = 0;
		double maxEnd = 0;
		for(int i = 0; i < num_postags; i++){
			if(v[i] > maxEnd){
				maxEnd = v[i];
				maxEndIndex = i;
			}
		}

		pred_seq = new int[len];
		int prev = maxEndIndex;
		for(int i = len - 1; i > 0; i--){
			prev = back_pointer[i][prev];
			pred_seq[i] = prev;
		}

		for(int i = 0; i < len; i++){
			s.getWordAt(i).setPosTag(inv_pos_tags.get(new Integer(pred_seq[i])));
			//System.out.printf("%s : %s\n", s.getWordAt(i).getLemme(), s.getWordAt(i).getPosTag());
		}

		return v[maxEndIndex];
	}

	/**
	 * Set the semi-supervised parameter \mu
	 */
	public void setMu(double _mu) {
		if (_mu < 0) {
			this.mu = 0.0;
		} else if (_mu > 1) {
			this.mu = 1.0;
		}
		this.mu = _mu;
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}
		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
		
		String trainingLogFileName = null;

		
		if (args.length > 3) {
			trainingLogFileName = args[3];
		}
		
		double mu = 0.0;
		
		if (args.length > 4) {
			mu = Double.parseDouble(args[4]);
		}
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);
		
		model.setMu(mu);
		System.out.println("Preparing HMM...");
		model.prepareMatrices();
		System.out.printf("Unsupervized learning... using mu = %.1f\n", mu);
		model.em();
		System.out.println("Generating predictions...");
		model.predict();
		System.out.println("Writing predictions output file...");
		model.outputPredictions(predictionFileName + "_" + String.format("%.1f", mu) + ".txt");
		System.out.println("Finished. Writing log files...\n\n");
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName + "_" + String.format("%.1f", mu) + ".txt");
		}
	}
}
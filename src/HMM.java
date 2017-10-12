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

	// emission matrix
	private Matrix B;

	// prior of pos tags
	private Matrix pi;

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
	private double smoothing_eps = 0.1;

	// number of iterations of EM
	private int max_iters = 10;
	
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

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
		System.out.printf("Reading in labeled corpus(%d) and preparing matrices.\n", labeled_corpus.size());
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
		num_postags = posCounter;
		num_words = vocabCounter;
		System.out.printf("Read in %d unique POS tags and %d unique vocab words.\n", num_postags, num_words);

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
		System.out.println("Populating A, B and pi");
		System.out.printf("POS Bigram size: %d\n", posBigrams.size());
		System.out.printf("Word to POS bigram size: %d\n", wordPOSBigram.size());

		int wordCount = 0;
		Set<IntPair> posPairs = posBigrams.keySet();
		for(IntPair pair : posPairs){
			double aij = (double)posBigrams.get(pair)/(double)posCount.get(pair.one);
			A.set(pair.one, pair.two, aij);
			//System.out.printf("%s, %s: %d  (%d)\n", inv_pos_tags.get(pair.one), inv_pos_tags.get(pair.two), posBigrams.get(pair), pair.hashCode());
		}

		Set<IntPair> pairs = wordPOSBigram.keySet();
		int oneMatch = 0;
		for(IntPair pair : pairs){
			double bij = (double)wordPOSBigram.get(pair)/(double)posCount.get(pair.two);
			B.set(pair.two, pair.one, bij);
			//System.out.printf("%d, %d: %d   (%d)\n", pair.one, pair.two, wordPOSBigram.get(pair), pair.hashCode());
			wordCount++;
			if(wordPOSBigram.get(pair).equals(1)){
				oneMatch++;
			}
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

		System.out.printf("%d word pos tag matches, %d had only one match\n", wordCount, oneMatch);
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
	}
	
	private Matrix posibilities;

	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
		viterbi(labeled_corpus.get(0));
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
	}
	
	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expection(Sentence s) {
		return 0;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
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
				sum += pi.get(0, i) * B.get(i, wIndex);
				if(temp > max){
					max = temp;
					maxPOSindex = i;
				}
			}
			result = Math.log(sum);
		}else{
			double prevResult = forwardHelper(s, position - 1);
			Word prevWord = s.getWordAt(position - 1);
			int prevPOSindex = pos_tags.get(prevWord.getPosTag());
			for(int i = 0; i < num_postags; i++){
				double temp = A.get(prevPOSindex, i) * B.get(i, wIndex);
				sum += temp;
				if( temp > max){
					max = temp;
					maxPOSindex = i;
				}
			}
			result = Math.log(sum) + prevResult;
		}
		word.setPosTag(inv_pos_tags.get(new Integer(maxPOSindex)));
		return result;
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
		return forwardHelper(s, s.length());
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
		return 0;
	}

	private Matrix vPaths;
	private Matrix curPprob;

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
		int wIndex = vocabulary.get(word.getLemme());
		for(int k = 0; k < num_postags; k++){
			double prob = pi.get(0, k) * B.get(k, wIndex);
			v[k] = Math.log(prob);
			back_pointer[0][k] = k;
		}
		for(int i = 1; i < len; i++){
			word = s.getWordAt(i);
			wIndex = vocabulary.get(word.getLemme());
			for(int j = 0; j < num_postags; j++){
				double max = 0;
				int maxPos = 0;
				for(int h  = 0; h < num_postags; h++){
					double temp = A.get(j, h) * B.get(h, wIndex);
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
			System.out.printf("%s : %s\n", s.getWordAt(i).getLemme(), inv_pos_tags.get(new Integer(pred_seq[i])));
		}

		return 0;
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
		
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);

		model.prepareMatrices();
		model.em();
		model.predict();
		model.outputPredictions(predictionFileName);
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName);
		}
	}
}

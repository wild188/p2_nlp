import java.util.Set;
import java.util.Hashtable;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import Jama.Matrix;

class HMM {
	/* Section for variables regarding the data */
	
	//
	private ArrayList<Sentence> labeled_corpus;
	
	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;
	
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
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;
	
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
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
	}
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
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
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s) {
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

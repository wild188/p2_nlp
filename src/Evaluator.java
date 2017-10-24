import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Evaluator {

	public static void main(String[] args) throws Exception {
		String groundTruthFileName = args[0];
		
		String predictionFileName = args[1];

		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(groundTruthFileName);
		
		ArrayList<Sentence> prediction_corpus = fh.readTaggedSentences(predictionFileName);
		
		if (labeled_corpus.size() != prediction_corpus.size()) {
			System.out.println("Two corpora have different sizes.");
			System.out.printf("Predicted length: %d Ground Truth length: %d\n", prediction_corpus.size(), labeled_corpus.size());
			checkDifs(labeled_corpus, prediction_corpus);
			System.exit(0);
		}
		
		double num_hits = 0;
		double total = 0;
		
		for (int i = 0; i < labeled_corpus.size(); ++i) {
			Sentence s1 = labeled_corpus.get(i);
			Sentence s2 = prediction_corpus.get(i);
			
			if (s1.length() != s2.length()) {
				System.out.println("Two sentences have different lengths. Location = " + i);
				System.exit(0);
			}
			
			for (int t = 0; t < s1.length(); ++t) {
				String tag1 = s1.getWordAt(t).getPosTag();
				String tag2 = s2.getWordAt(t).getPosTag();
				if ((tag1 != null && tag1.equals(tag2)) ||
						(tag2 != null && tag2.equals(tag1)) ) {
					num_hits++;
					//System.out.println(tag1 + " " + tag2);
				}else{
					//System.out.println(tag1 + " " + tag2);
				}
				total++;
			}
			
			//break;
		}
		//System.out.print("Accuracy = ");
		System.out.println("Accuracy = " + num_hits / total);
	}

	private static void checkDifs(ArrayList<Sentence> labeled_corpus, ArrayList<Sentence> prediction_corpus){
		int missed = 0;
		for(int i = 0; i < labeled_corpus.size(); i++){
			if(labeled_corpus.get(i).toString().equals(prediction_corpus.get(i - missed).toString())){
				System.out.println(labeled_corpus.get(i).toString());
				missed++;
			}
		}
	}
}

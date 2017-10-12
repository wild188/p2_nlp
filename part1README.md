Billy DeLucia
CSE 498 NLP Project 2  Phase 1

This project is to develop an HMM model trained on a POS tagged corpus.
Currently I am generating the predictions based on the base A, B, and pi. 
Not handling unseen words, only substituted with 0.1, obviously this will be improved.
Not currently using the EM model. Considering adding some special cases for unseen words
in the middle of the sentence with capital first letter -> NNP, abd maybe weight special 
symbols like commas and periods more. 

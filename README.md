## AntiPlagiarismMl
### An anti-plagiarism utility that compares two texts of Python programs and gives an estimate of their similarity. 
### The problem is solved using machine learning methods.

The program combines several approaches for comparing programs:   
text-based,  
token-based,  
binary-based;  
The metrics are the _Damerau-Levenshtein distance_ and _LCS_,  
both algorithms have complexity __O(n^2)__ and are __recursive__.  
First of all, the binary representation of programs is compared, then the code is tokenized and compared in the same way.  
If there is a difference between the evaluation of the binary representation and the tokenized one, an additional lexeme check is performed.  
_Therefore, please, when running the algorithm on a large set of files, keep in mind that it may take a lot of time._  
When developing the algorithm and training the model, the following results were obtained:  
Calculation time for _5_ pairs of files: _10_ seconds;  
Calculation time for _312_ pairs of files: _2052_ seconds;  
Time required to train the model: _5831_ seconds;  
The accuracy of the resulting model: 0.9948.  
Sources:   
A Survey on Software Clone Detection Research,
Chanchal Kumar Roy and James R. Cordy,
September 26, 2007  

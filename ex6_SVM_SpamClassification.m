% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs

% Data note: Data contains raw email text. The porterStemmer function (not
% included here) parses and processes this text, converting it to a word stem vector. 
% A dictionary(vocabList) with spam-associated words is compared to the email vector 
% resulting in binary(0/1) vector indicating if a dictionary word (by index) was present.
% This binary vector is used along with the spam/not spam binary labels in y to 
% train a SVM linear classifier.

clear ; close all; clc

% Functions...

function vocabList = getVocabList()
	% reads the fixed vocabulary list in vocab.txt and returns a cell array of 
	% the words in vocabList.
	% Read the fixed vocabulary list...
	fid = fopen('vocab.txt');
	% Store all dictionary words in cell array vocab{}
	n = 1899;  % Total number of words in the dictionary
	% For ease of implementation, we use a struct to map the strings => integers
	% Note: In practice, use a hashmap.
	vocabList = cell(n, 1);
	for i = 1:n
		% Word Index (can ignore since it will be = i)
		fscanf(fid, '%d', 1);
		% Actual Word
		vocabList{i} = fscanf(fid, '%s', 1);
	end
	fclose(fid);
end

function word_indices = processEmail(email_contents)
	% preprocesses the body of an email and returns a list of indices of the 
	% words contained in the email. 
	% Load Vocabulary...
	vocabList = getVocabList();
	% Initialize values...
	word_indices = [];

	% Preprocessing...
	% Find the Headers ( \n\n and remove ): Uncomment the following lines if you are 
	% working with raw emails with the full headers:
	% hdrstart = strfind(email_contents, ([char(10) char(10)]));
	% email_contents = email_contents(hdrstart(1):end);
	% Lower case...
	email_contents = lower(email_contents);
	% Strip all HTML: Looks for any expression that starts with < and ends with > 
	% and replace and does not have any < or > in the tag it with a space...
	email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
	% Handle Numbers: Look for one or more characters between 0-9...
	email_contents = regexprep(email_contents, '[0-9]+', 'number');
	% Handle URLS: Look for strings starting with http:// or https://
	email_contents = regexprep(email_contents, ...
							   '(http|https)://[^\s]*', 'httpaddr');
	% Handle Email Addresses...
	% Look for strings with @ in the middle
	email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
	% Handle $ sign...
	email_contents = regexprep(email_contents, '[$]+', 'dollar');
	
	% Tokenizing...
	% Output the email to screen as well
	fprintf('\n==== Processed Email ====\n\n');
	% Process file
	l = 0;
	while ~isempty(email_contents)
		% Tokenize and also get rid of any punctuation
		[str, email_contents] = ...
		   strtok(email_contents, ...
				  [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
	   	% Remove any non alphanumeric characters
		str = regexprep(str, '[^a-zA-Z0-9]', '');
		% Stem the word 
		% (the porterStemmer sometimes has issues, so we use a try catch block)
		try str = porterStemmer(strtrim(str)); 
		catch str = ''; continue;
		end;
		% Skip the word if it is too short
		if length(str) < 1
		   continue;
		end
		% Look up word (from str) in the dictionary (vocabList). If a match
		% exists, add the vocabList index of the word to the word_indices vector. 
		lengthEmail = length(str);
		n = 1899;  % Total number of words in the dictionary
		% looping over dictionary, to identify email words (str) matching dictionary...
		for d = 1:n
			if 	strcmp(str, vocabList{d})
					word_indices = [word_indices; d];
			end
		end
		% Print to screen, ensuring that the output lines are not too long
		if (l + length(str) + 1) > 78
			fprintf('\n');
			l = 0;
		end
		fprintf('%s ', str);
		l = l + length(str) + 1;
	end
	% Print footer
	fprintf('\n\n=========================\n');
end

function sim = linearKernel(x1, x2)
	% returns a linear kernel between x1 and x2
	% and returns the value in sim
	% Ensure that x1 and x2 are column vectors
	x1 = x1(:); x2 = x2(:);
	% Compute the kernel
	sim = x1' * x2;  % dot product
end

function [model] = svmTrain(X, Y, C, kernelFunction, ...
                            tol, max_passes)
	% trains an SVM classifier and returns trained model. X is the matrix 
	% of training examples. Each row is a training example, and the jth column 
	% holds the jth feature. Y is a column matrix containing 1 for positive 
	% examples and 0 for negative examples.  C is the standard SVM regularization 
	% parameter.  tol is a tolerance value used for determining equality of 
	% floating point numbers. max_passes controls the number of iterations
	% over the dataset (without changes to alpha) before the algorithm quits.
	% Note: This is a simplified version of the SMO algorithm for training SVMs. 
	if ~exist('tol', 'var') || isempty(tol)
		tol = 1e-3;
	end
	if ~exist('max_passes', 'var') || isempty(max_passes)
		max_passes = 5;
	end
	% Data parameters
	m = size(X, 1);
	n = size(X, 2);
	% Map 0 to -1
	Y(Y==0) = -1;
	% Variables
	alphas = zeros(m, 1);
	b = 0;
	E = zeros(m, 1);
	passes = 0;
	eta = 0;
	L = 0;
	H = 0;
	% Pre-compute the Kernel Matrix since our dataset is small
	% We have implemented optimized vectorized version of the Kernels...
	if strcmp(func2str(kernelFunction), 'linearKernel')
		% This is equivalent to computing the kernel on every pair of examples
		K = X*X';
	elseif strfind(func2str(kernelFunction), 'gaussianKernel')
		% This is equivalent to computing the kernel on every pair of examples
		X2 = sum(X.^2, 2);
		K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
		K = kernelFunction(1, 0) .^ K;
	else
		% Pre-compute the Kernel Matrix
		K = zeros(m);
		for i = 1:m
			for j = i:m
				 K(i,j) = kernelFunction(X(i,:)', X(j,:)');
				 K(j,i) = K(i,j); %the matrix is symmetric
			end
		end
	end
	% Train
	fprintf('\nTraining ...');
	dots = 12;
	while passes < max_passes,
		num_changed_alphas = 0;
		for i = 1:m,
			% Calculate Ei = f(x(i)) - y(i) using (2). 
			% E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
			E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
			if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0)),
				%In this simplified code, we select i and j randomly.
				j = ceil(m * rand());
				while j == i,  % Make sure i \neq j
					j = ceil(m * rand());
				end
				% Calculate Ej = f(x(j)) - y(j) using (2).
				E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);
				% Save old alphas
				alpha_i_old = alphas(i);
				alpha_j_old = alphas(j);
				% Compute L and H by (10) or (11). 
				if (Y(i) == Y(j)),
					L = max(0, alphas(j) + alphas(i) - C);
					H = min(C, alphas(j) + alphas(i));
				else
					L = max(0, alphas(j) - alphas(i));
					H = min(C, C + alphas(j) - alphas(i));
				end
				if (L == H),
					% continue to next i. 
					continue;
				end
				% Compute eta by (14).
				eta = 2 * K(i,j) - K(i,i) - K(j,j);
				if (eta >= 0),
					% continue to next i. 
					continue;
				end
				% Compute and clip new value for alpha j using (12) and (15).
				alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
				% Clip
				alphas(j) = min (H, alphas(j));
				alphas(j) = max (L, alphas(j));
				% Check if change in alpha is significant
				if (abs(alphas(j) - alpha_j_old) < tol),
					% continue to next i. 
					% replace anyway
					alphas(j) = alpha_j_old;
					continue;
				end
				% Determine value for alpha i using (16). 
				alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
				% Compute b1 and b2 using (17) and (18) respectively. 
				b1 = b - E(i) ...
					 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
					 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
				b2 = b - E(j) ...
					 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
					 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
				% Compute b by (19). 
				if (0 < alphas(i) && alphas(i) < C),
					b = b1;
				elseif (0 < alphas(j) && alphas(j) < C),
					b = b2;
				else
					b = (b1+b2)/2;
				end
				num_changed_alphas = num_changed_alphas + 1;
			end
		end
		if (num_changed_alphas == 0),
			passes = passes + 1;
		else
			passes = 0;
		end
		fprintf('.');
		dots = dots + 1;
		if dots > 78
			dots = 0;
			fprintf('\n');
		end
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		end
	end
	fprintf(' Done! \n\n');
	% Save the model
	idx = alphas > 0;
	model.X= X(idx,:);
	model.y= Y(idx);
	model.kernelFunction = kernelFunction;
	model.b= b;
	model.alphas= alphas(idx);
	model.w = ((alphas.*Y)'*X)';
end


function pred = svmPredict(model, X)
	% returns a vector of predictions using a trained SVM model (svmTrain). 
	% X is a mxn matrix where there each example is a row. model is a svm 
	% model returned from svmTrain. predictions pred is a m x 1 column of 
	% predictions of {0, 1} values.
	% Check if we are getting a column vector, if so, then assume that we only
	% need to do prediction for a single example
	if (size(X, 2) == 1)
		% Examples should be in rows
		X = X';
	end
	% Dataset 
	m = size(X, 1);
	p = zeros(m, 1);
	pred = zeros(m, 1);
	if strcmp(func2str(model.kernelFunction), 'linearKernel')
		% We can use the weights and bias directly if working with the 
		% linear kernel
		p = X * model.w + model.b;
	elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
		% Vectorized RBF Kernel
		% This is equivalent to computing the kernel on every pair of examples
		X1 = sum(X.^2, 2);
		X2 = sum(model.X.^2, 2)';
		K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
		K = model.kernelFunction(1, 0) .^ K;
		K = bsxfun(@times, model.y', K);
		K = bsxfun(@times, model.alphas', K);
		p = sum(K, 2);
	else
		% Other Non-linear kernel
		for i = 1:m
			prediction = 0;
			for j = 1:size(model.X, 1)
				prediction = prediction + ...
					model.alphas(j) * model.y(j) * ...
					model.kernelFunction(X(i,:)', model.X(j,:)');
			end
			p(i) = prediction + model.b;
		end
	end
	% Convert predictions into 0 / 1
	pred(p >= 0) =  1;
	pred(p <  0) =  0;
end

function x = emailFeatures(word_indices)
	% takes in a word_indices vector and produces a binary vector equal to
	% value 1 if a dictionary (vocabList) word was present in the email,
	% 0 if not.
	n = 1899; % Total number of words in the dictionary
	% Initialize values...
	x = zeros(n, 1);
	% Looping over the dictionary word indices, set = 1 where present
	% in email (word_indices)...
	for w = 1:length(word_indices),
		x(word_indices(w)) = 1;
	end
end

% Script...

%% ==================== Part 1: Email Preprocessing ====================
%  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
%  to convert each email into a vector of features. In this part, you will
%  implement the preprocessing steps for each email. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given email.

fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================
%  Now, you will convert each email into a vector of features in R^n. 
%  You should complete the code in emailFeatures.m to produce a feature
%  vector for a given email.

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
%

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');


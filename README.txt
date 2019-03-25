You are to implement (or find the code for) six algorithms. The first two are clustering algorithms:

k-means clustering
Expectation Maximization
You can choose your own measures of distance/similarity. Naturally, you'll have to justify your choices, 
but you're practiced at that sort of thing by now.

The last four algorithms are dimensionality reduction algorithms:

PCA
ICA
Randomized Projections
Any other feature selection algorithm you desire
You are to run a number of experiments. Come up with at least two datasets. If you'd like (and it makes a lot of sense in this case) 
you can use the ones you used in the first assignment.

Run the clustering algorithms on the datasets and describe what you see.
Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from 
assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms 
(you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms 
as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
What to Turn In
You must submit:

A file named README.txt that contains instructions for running your code
your code (link only in the README.txt)
a file named yourgtaccount-analysis.pdf that contains your writeup.
The file yourgtaccount-analysis.pdf should contain: 

a discussion of your datasets, and why they're interesting: If you're using the same datasets as before at least briefly remind us of 
what they are so we don't have to revisit your old assignment write-up.
explanations of your methods: How did you choose k?
a description of the kind of clusters that you got.
analyses of your results. Why did you get the clusters you did? Do they make "sense"? If you used data that already had labels 
(for example data from a classification problem from assignment #1) did the clusters line up with the labels? Do they otherwise line up 
naturally? Why or why not? Compare and contrast the different algorithms. What sort of changes might you make to each of those algorithms 
to improve performance? How much performance was due to the problems you chose? Be creative and think of as many questions you can, and 
as many answers as you can. Take care to justify your analysis with data explicitly.
Can you describe how the data look in the new spaces you created with the various algorithms? For PCA, what is the distribution of 
eigenvalues? For ICA, how kurtotic are the distributions? Do the projection axes for ICA seem to capture anything "meaningful"? 
Assuming you only generate k projections (i.e., you do dimensionality reduction), how well is the data reconstructed by the randomized 
projections? PCA? How much variation did you get when you re-ran your RP several times (I know I don't have to mention that you might 
want to run RP many times to see what happens, but I hope you forgive me)?
When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA, PCA, and RP, did you get 
the same clusters as before? Different clusters? Why? Why not?
When you re-ran your neural network algorithms were there any differences in performance? Speed? Anything at all?
It might be difficult to generate the same kinds of graphs for this part of the assignment as you did before; however, you should come 
up with some way to describe the kinds of clusters you get. If you can do that visually all the better. 

Jenny@LAPTOP-GCDICEA9 MINGW64 ~/Dropbox/gt/spring 2019/cs 4641/p3
$ python -W ignore analysis.py spam
NNonKM time: 1444.14600015
km done

Jenny@LAPTOP-GCDICEA9 MINGW64 ~/Dropbox/gt/spring 2019/cs 4641/p3
$ python -W ignore analysis.py spam
NNonEM time: 1404.73099995
em done

Jenny@LAPTOP-GCDICEA9 MINGW64 ~/Dropbox/gt/spring 2019/cs 4641/p3
$ python -W ignore analysis.py spam
PCA
[0.95634767 0.99758655 0.99986374 0.99991935 0.99993336 0.99994337
 0.99995181 0.99995885 0.9999646  0.99996932 0.99997317 0.99997675
 0.99997989 0.99998223 0.99998411 0.99998584 0.99998741 0.99998854
 0.99998958 0.99999044 0.99999115 0.9999918  0.99999244 0.99999302
 0.99999355 0.99999401 0.99999444 0.99999484]
NNonEMwPCAtr time: 1048.81099987
NNonKMwPCAtr time: 1059.88300014
NNwPCAr time: 1047.16499996
pca done

Jenny@LAPTOP-GCDICEA9 MINGW64 ~/Dropbox/gt/spring 2019/cs 4641/p3
$ python -W ignore analysis.py spam
PCA
[0.95634767 0.99758655 0.99986374 0.99991935 0.99993336 0.99994337
 0.99995181 0.99995885 0.9999646  0.99996932 0.99997317 0.99997675
 0.99997989 0.99998223 0.99998411 0.99998584 0.99998741 0.99998854
 0.99998958 0.99999044 0.99999115 0.9999918  0.99999244 0.99999302
 0.99999355 0.99999401 0.99999445 0.99999484]
[95.6 99.7 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9
 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9 99.9]
pca done

$ python -W ignore analysis.py spam
[28.53217623 50.49128201 51.99409248 50.70276237 43.27828467 46.60062679
 48.34888992 49.62575015 30.37071374 24.89085493 25.29397596  3.137271
 45.89291394 51.84195645 15.16241179 39.51095181 13.46549205 39.98202615
 51.76810973 25.83301135 34.19832668 51.30070238 24.20095957 51.31297765
 40.62479383 46.06106358  6.61094008 19.53444371  6.32994582 19.32921146
 48.74420002 47.87064115 35.14186637 35.24526746 30.01731676 49.10422767
 24.25183111 29.96957571 32.54228101 52.01207548 50.51842626 21.92693035
 47.90206693 39.204565   37.59040369 35.72262707 49.28554952 16.9364577
 30.54595244 51.98887993 34.50748985 25.30436576 51.48298481 51.71256918
 51.8602075  50.20526858 22.38133929]
Kurtosis: 52.012075481962
NNonEMwICAtr time: 2099.65499997
NNonKMwICAtr time: 2086.67400002
NNwICAtr time: 72160.5109999
ica done

NNwRP time: 712.198999882
randproj done
NNwKB time: 1109.75899982
NNwPCA time: 1223.16200018
NNwICA time: 2486.98800015

Despite musical information being something that is so naturally interpreted by us, the differences
that separate musical works is something that is not as intuitively quantified. Here we will utilize methods
of signal processing, statistics, and linear algebra to classify the difference between musical works.
Classification will be explored on three levels: between any three artists, three artists within the same
genre, and between works in three different genres.  
Though we can easily recognize difference between songs, artists, and genres, the displacement
over time information of a speaker creating music couldn’t be further removed from the stylistic
properties like pitch, rhythm, tempo, phrasing and timbre that vary so wildly between music. These
musical can be more easily thought of as a relationship of what frequencies are present in a signal at a
given time. For this reason, we will utilize spectrograms extensively to get a handle on this information.
However, finding patterns between spectrograms remains non-trivial. Principal Component
Analysis(PCA) is an incredibly useful tool for representing these complicated relationships in a way that
best highlights differences and will be used in conjunction Linear Discriminant Analysis(LDA) to make
some claim as to what should actually be classified as what.  
The process of creating a classifier follows a process of training an algorithm with a set of
dedicated training data and a set of test data that remains unseen during the training phase.  

This codes functionality is split into two parts: hw4setup.m and hw4Group3.m. 
hw4Setup will take a directory of mp3 files and processes them into spectrograms of 5 second log clips. 
These spectrograms are croped and downsample to a series of 100 by 100 images.  
hw4Group3.m will take in saved collections of these spectrograms and train a classifier using Linear Discriminat Analysis.


custom mean shift algorithm. Used Sentdex's great tutorial video series as guidance (see below for
links) but tried to make my own version that I think might run faster as I do not make a large
iterable just to take the average vector when calculating the new average vector by weighting
vectors closer to the current cluster.

Note: no gurantees on how consistent it works but eyeballed a handful of times using sklearn
blobs and seems to do well when checking it visually on a 2-d dataset.

Sentdex references:
Videos:
https://www.youtube.com/watch?v=P-iAd8b7zl4&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=41
https://www.youtube.com/watch?v=k1alPDpSGBE&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=42

Website:
https://pythonprogramming.net/weighted-bandwidth-mean-shift-machine-learning-tutorial/
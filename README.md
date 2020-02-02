Face recognition project to identify different actors in Koibumi by Tanaka Kinuyo (1953) and count the number of screenshots in which the different actors appear. The code is developed using a mixture of Open CV2 libraries and customized deep neural networks. The pipeline is divided in two parts. A first code trains the networks on two sets of about 50 images for each actor. A second code scans through a video, uses the CV2 libraries to identify the presence of a face and use the previously trained network to recognize if the identified face belogns to one of the two actors. 
<p>
<p>
<img src="https://github.com/ecancellieri/Face_Recognition_Koibumi/blob/master/Yoshiko_Kuga.png" width="400">
<img src="https://github.com/ecancellieri/Face_Recognition_Koibumi/blob/master/Masayuki_Mori.png" width="400">
<p>
<p>
<p>
Two exapmles of the output of the code recognizing Masayuki Mori and Yoshiko Kuga in Kinuyo Tanaka's "Koibumi" (1953)

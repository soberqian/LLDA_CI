# LLDA_CI
LinkLDA tool for acquiring competitive intelligence 

# CVB for LinkLDA
We use Collapsed Variational Bayesian (CVB) Inference for learning the parameters of Link-LDA.
The input file likes:<br />
```java
Volkswagen Golf--BMW	jhend925 correct gti heat seat include trim
Kia Soul--Ford Escape--Toyota RAV4	car_man current lease number Soul Exclaim premium package market quote pathetic offer walk dealership contact EXACT vehicle
```
The products and their contents are used table to split.

The following code is to call the LinkLDA algorithm for processing these documents:<br />
```java
		LinkLDACVB linklda = new LinkLDACVB("test/data/cardatatest.txt", "gbk", "\t", "--", 15, 0.1,
				0.01,0.01, 20, 50, "test/output/");
		linklda.CVBInference();
```

The contents of topic-product distribution like: <br /> 
```
Topic:1
Lexus IS 200t :0.82801393728223
GMC Acadia :0.7883767535070141
Saturn L300 :0.775535590877678
Lexus NX 300h :0.7683754512635379
Porsche Cayenne :0.7153568853640953
Audi R8 :0.610393466963623
Oldsmobile Alero :0.5796109993293093
...

Topic:2
BMW X6 :0.32407809110629066
Lincoln Continental :0.255003599712023
Audi A5 :0.2263959390862944
Ford Edge :0.1896140350877193
Cadillac ATS-V :0.1740510697032436
Pontiac G5 :0.1566591422121896
Lexus NX 300 :0.13647570703408266
Volkswagen Tiguan :0.13225579761068165
...
```
The contents of topic-word distribution like: <br /> 

```java
Topic:1
drive :0.08731467480511887
awd :0.06541822490968394
post :0.054768834886097705
time :0.03145971080386051
base :0.030427371975043478
rate :0.029014697788241225
high :0.027765024469146922
door :0.02695002013060716
show :0.024179005379571968
...

Topic:2
person :0.028071566434497267
miles/year :0.02691409539297589
hood :0.01939053362308692
article :0.015918120498522776
max :0.014760649457001397
console :0.013313810655099673
massachusetts :0.013313810655099673
rubber :0.013024442894719327
section :0.010709500811676568
...
```

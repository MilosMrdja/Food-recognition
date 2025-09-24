## YOLO 

1. ###### Pronalazenje dataseta sa odgovarajucim bounding boxovima - moj dataset oko 7k slika



*  	Proveriti koliko su tacni bounding boxovi
*  	Proci kroz dataset videti kvalitet slika, duplikate



##### 2\. Predprocesiranje dataseta

*  	Svaki set je namesten drugacije,treba da vidimo kako je nas organizovan I da ga prilagodimo nasim potrebama
* &nbsp;	Pisao sam sskirpte za pronalazak duplikata, prolazak kroz dataset I gledanje oznacene bounding boxove
*  	Gledao koliko koja slika ima labela, vise objekata na jednoj slici
* &nbsp;	**KONVERTOVANJE**
* 		**INPUT\_DIR = data/ test**

				**/ train**

				**/ valid**

		svaki taj podfolder sadrzi images I labels foldere, labels sadzri <ime>.txt u kome se nalaze bounding boxovi 

&nbsp;		pr. **17 0.06875 0.4609375 0.1375 0.5390625**. <class\_id>  <x\_center>  <y\_center>  <width>  <height>. 

&nbsp;		ime txt fajla je zapravo slika koja je sacuvana u images folderu pod tim imenom, tako svaka slika moze da ima vise labela

&nbsp;		konvertovanje je ovde konrketno bilo u format da sve klase budu 0, jer yolo u nasem slucaju treba samo da prepozna objekat

##### 3\. Treniranje yolo modela 

* cuda radi samo na nvidia, a ROCm radi samo na linuxu za amd, ja amd i windows -> morao sam da koristim google colab koji ima free tier za koriscenje gpu-s
* zipovane datasetove sam upload na drive, mount google drive sa dirve, unzip foldere i pokretao .ipnyb fajlove da istreniraju model, model kasnije cuvao I dodao u projekat
* Bitno je koliko slika ima labela, eksepiremnt sa labels max = 2

##### 

##### 4\. Testiranje yolo modela

* Ucitamo istrenirani yolo model, pokrenemo nad test podacima i vidimo metrike

##### 

##### 5\. Metrike

1. precision - ≥ 0.80 (vrlo dobro), 0.90+ odlično - koliko je stvarno ispravno
2. Recall - ≥ 0.70 (vrlo dobro), 0.85+ odlično - Koliki procenat stvarnih objekata je pronađen.
3. mAP@0.50 - ≥ 0.50 (prihvatljivo), 0.70+ dobro, 0.85+ vrhunsko - 
4. mAP@0.50:0.95 - ≥ 0.30 (prihvatljivo), 0.50+ dobro, 0.60+ vrhunsko - Striktnija provera od mAP@0.50



## CNN - RESNET

1. ##### Pronalazenje dataseta - moj dataset oko 15k slika

* Potreban je dovoljno velik dataset bez puno duplikata, augmentacija se radi u trenrianju modela
* Prov sam radio na datasetu koji ima vecinu duplikata i nisam dobio dovoljno dobre metrike

##### 2\. Predprocesiranje dataseta

* **KONVERTOVANJE**
* 	**INPUT\_DIR = images/** podfolderi od 0 do 81(klase) u svakom folderu slike
* 		    **Meata**/ training, test, validation .txt koji imaju nazive slika
* &nbsp;	Treba predprocesirati da budu u obliku OUTPUT\_DIR/ testing
* &nbsp;							   training
* &nbsp;							   validation	
* &nbsp;								I u svakom ovoom podfolderu folderi koji predstavljaju klase i unutra slike za tu klasu **ImageFolder Struktura**	
* 

##### **3. Trenianje CNN-RESNET Modela**

* Priprema podatak - ImageFolder
* Koristi se torchvision.models.resnet18 sa pretreniranom težinom
* **Gubitak - CrossEntropyLoss (standardni za multi-class klasifikaciju). meri koliko su predikcije daleko od tačnih klasa**
* **Optimizator - SDK, Adam koji na osnovu gradijenata menja težine da smanji loss i tako „uči“ model.**
* **Adam, SDK, 0.001, 0.0001, batch =32/64, Stopper**
* **Batch – broj uzoraka koji mreža obrađuje pre jedne “optimizacione” iteracije**



##### 4\. Testiranje CNN-RESNET modela

* Ucitamo istrenirani model, pokrenemo nad test podacima i vidimo metrike
* 
* 
**##### 5\. Metrike**

* **Accuracy = (broj ispravno klasifikovanih slika) ÷ (ukupan broj test slika), koliko je u proseku tacno slika pogodio**




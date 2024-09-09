# project3

Αλεξία Τοπαλίδου: 1115201600286	

Στέφανος Μπακλαβάς: 1115201700093

Ερωτημα Α: Πρόβλεψη χρονοσειρών με τη χρήση LSTM (Long Short-Term Memory) μοντέλου.
 
 Αρχικα για καθε χρονοσειρα γινεται train και προβλεψη για την συγκεκριμενη χρονοσειρα και στη συνεχεια αφου τελειωσει
 αυτη η διαδικασια φορτώνεται το μοντελο που εχει εκπαιδευτει με ολες τις χρονοσειρες και γινεται προβλεψη με αυτο.
 Πρόγραμμα :forecast.py
 run: ./forecast.py


 Ερωτημα Β.
 
 The script loads a time series dataset, processes it through an autoencoder to detect anomalies based 
 reconstruction error and generates plots that visualize the reconstruction loss and detected anomalies
 
 Ανίχνευση ανωμαλιών σε χρονοσειρές.
 
 Autoencoder for anomaly detection. Συμπιεζει την χρονοσειρα και προσπαθει να την ανακατασκευασει και βρισκει τις 
 ανωμαλιες που θα δημιουργησουν προβλημα στην ανακατασκευη της χρονοσειρας με βαση καποιο threshld που θετουμε εμεις
  
 
 detect.py
 
 python3 detect.py -d ./dataset.csv -n 4 -m 0.0015
 n= αριθμος χρονοσειρων 
 m= thershold για ανιχνευση ανωμαλιων
 


 Ερωτημα Γ,
 
 This Python script reduces the dimensionality of two input time series datasets (a dataset and a query set) using a pre-trained autoencoder model. It reads the datasets, compresses them with the autoencoder, and writes the reduced time series to output files
 
 python reduce.py -d dataset.csv -q queryset.csv -od compressed_dataset.csv -oq compressed_query.csv
 
 The code is designed to compress time series data using a pre-trained autoencoder, reducing its dimensionality while maintaining the structure of the data. This can be useful for tasks like anomaly detection, query matching, or reducing storage and computational costs for large datasets.
 
 Ανακατασκευη χρονοσειρας 
 reduce.py
 
 python3 reduce.py -d ./dataset.csv -q ./query.csv -od ./compressed_dataset.csv -oq ./compressed_queryset.csv



Στον συγκεκριμένο φάκελο υπάρχουν τα εξής.

1.Μοντέλα:

α.Το εκπαιδευμένο μοντέλο για το ερώτημα Α: my_model
β.Το εκπαιδευμένο μοντέλο για το ερώτημα Β: autoencoder_3b
γ.To εκπαιδευμένο μοντέλο για το ερώτημα Γ: autoencoder_3c (δεν χρησιμοποιείται κάπου)
δ.Το εκπαιδευμένο μοντέλο για το ερώτημα Γ: encoder_3c


2.Colab Notebooks:

Η εκπαίδευση και οι δοκιμές για το ερώτημα Β και Γ έγινε με τη χρήση Google Colab και τα Notebooks είναι τα project_3b.ipynb και project_3c.ipynb


3.Αρχεία .py:

α. forecast.py το οποίο χρησιμοποιήθηκε και για την εκπαίδευση αλλά και για τη φόρτωση του καλύτερου μοντέλου στην εξέτση για το ερώτημα Α.

b. detect.py το οποίο φορτώνει το καλύτερο μοντέλο για το ερώτημα β και παράγει τα γραφήματα στον φάκελο B_plots

c. reduce.py το οποίο φορτώνει το καλυτερο μοντέλο για το ερώτημα Γ και παράγει τα συμπιεσμένα αρχεία


4.Οι φάκελοι outputs_project2 και compressed_outputs_project2 που περιέχουν τα αποτελέσματα για όλα τα ερωτήματα του 2ου project με το αρχικό και το συμπιεσμένο dataset αντίστοιχα.Για οικονομία χρόνου χρησιμοποιήθηκαν οι πρώτες 50 χρονοσειρές για dataset και οι 3 επόμενες για queryset από το αρχείο που μας είχε δοθεί.


5.Δύο scalers για τα ερωτήματα B και Γ που έχουν γίνει fit με το σύνολο του dataset των 360 χρονοσειρών.


6.Αρχεία.csv 


7.Η αναφορά της εργασίας με τίτλο Project3_report.pdf






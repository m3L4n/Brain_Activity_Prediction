We Have to create a script that permit to vizalize the EEG DATA with library MNE

-> MNE = Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.

To vizualize data we have to dl all the file (data) of EGG

1 first step :

- copy past this command in our terminal wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/ ( it take a long time)

-> apres on va regarder le tuto de mne sur comment afficher toute sa data ( eeg)
on va afficher le psd ( puissance de la densite spectrale du signal afin de visualizer quel frequence peut etre bonne + connaissance des frequences )

# Bande Delta (0.5-4 Hz) : Généralement associée au sommeil profond.

    # Bande Theta(4-8 Hz): Souvent associée à la somnolence et à la relaxation.
    # Bande Alpha(8-13 Hz): Typiquement liée à l'état de relaxation, surtout avec les yeux fermés.
    # Bande Beta(13-30 Hz): Associée à l'activité cognitive, l'éveil et la concentration.
    # Bande Gamma(30-100 Hz): Liée à des processus cognitifs supérieurs comme l'attention et la mémoire de travail.
    # Le pic autour de 50 Hz est probablement un bruit de l'alimentation électrique(50 Hz est la fréquence du courant alternatif en Europe, 60 Hz aux États-Unis).
    # Les hautes fréquences au-dessus de 40 Hz peuvent contenir du bruit musculaire ou d'autres artefacts non liés au signal EEG.

ensuite on filte de 8 a 40 hz psk c'est les channels qui nous interresse
-> on affiche on voit bien du coup que de 9 a 40 c'est ce qu il nous faut
et apres on va filter et faire une moyenne des data afin d'une ligne plus clean , et bien voir exactement la courbe de notre signal

-> apres avoir filter et biensur concatener tout nos .edf ( avec une frequence ( d'enregistrement par seconde) a 160 qui defini dans le le physionet ou on dispose de la data )

on va pouvoir separer nos raw en epoch ( separer les differents mouvements par rapport au signaux) et effectue une classification avec le csp ( voir pipeline_traitement pour la deuxieme partie du sujet )

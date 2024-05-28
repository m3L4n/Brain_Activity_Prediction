-> le sujet nous demande d'utiliser un playback afin d'avoir l impression d'un procces en temps real

->>>> POUR LA PREDICTION
-> du coup ca veut dire qu on va charger au fur et a mesure des chunk et ecrire le resultat dans le terminal avec un delai de 2s

-> apres avoir mit en place un playback on va pvr commencer par utiliser l'algo csp fournit par sclearn afin de se familariser avec cette algorithme et comprendre ce qu il faut faire.

EN first on decoupe nos data

-> il faut decouper ses data en EPOH c 'est a dire quel action a ete effectue a quel moment

donc
tasks = [
('Baseline, eyes open', 60),
('Baseline, eyes closed', 60),
('Task 1 (open and close left or right fist)', 120),
('Task 2 (imagine opening and closing left or right fist)', 120),
('Task 3 (open and close both fists or both feet)', 120),
('Task 4 (imagine opening and closing both fists or both feet)', 120),
('Task 1', 120),
('Task 2', 120),
('Task 3', 120),
('Task 4', 120),
('Task 1', 120),
('Task 2', 120),
('Task 3', 120),
('Task 4', 120),
]

comme dit dans le sujet
Each subject performed 14 experimental runs: two one-minute baseline runs (one with eyes open, one with eyes closed), and three two-minute runs of each of the four following tasks:

essentiel de cree des epoch car ca permet de garder ce qui est vraiment interresant ici ( quand le sujet fait une action)

Préparation des Données : Les données EEG brutes sont complexes et contiennent des informations continues et variées. Les epochs permettent de structurer ces données en segments uniformes, ce qui est essentiel pour une analyse cohérente.

Extraction des Caractéristiques : Les algorithmes comme le CSP (Common Spatial Patterns) extraient des caractéristiques à partir des epochs. Ces caractéristiques sont ensuite utilisées pour entraîner des modèles de classification.

Uniformité des Échantillons : Les modèles de classification comme les SVC (Support Vector Classifiers) ou les LDA (Linear Discriminant Analysis) nécessitent des échantillons de taille fixe pour l'entraînement. Les epochs assurent que chaque segment de données utilisé pour l'entraînement a la même durée et structure.

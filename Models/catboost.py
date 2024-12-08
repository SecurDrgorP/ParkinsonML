from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
class CatBoostScratch:
    def _init_(self, n_estimators=500, learning_rate=0.1, max_depth=3, l2_reg=0.1, early_stopping_rounds=50):
        """
        Initialiser les paramètres de CatBoost.
        
        Paramètres :
        - n_estimators : Nombre total d'arbres (itérations de boosting).
        - learning_rate : Taux d'apprentissage pour mettre à jour les prédictions.
        - max_depth : Profondeur maximale des arbres.
        - l2_reg : Régularisation L2 pour réduire le surapprentissage.
        - early_stopping_rounds : Nombre d'itérations sans amélioration avant d'arrêter l'entraînement.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []
        self.base_prediction = None
        self.feature_importances_ = None

    def _log_loss_gradient(self, y_true, y_pred):
        """
        Calculer le gradient de la fonction log-loss par rapport aux prédictions.
        
        Paramètres :
        - y_true : Étiquettes réelles.
        - y_pred : Prédictions actuelles (logits).
        
        Retourne :
        - Gradient de log-loss.
        """
        sigmoid = 1 / (1 + np.exp(-y_pred))  # Appliquer la fonction sigmoid
        return y_true - sigmoid

    def _check_balance(self, y):
        """
        Vérifier l'équilibre des classes dans les étiquettes y.
        """
        pos_ratio = np.mean(y)
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            print("Attention : Dataset déséquilibré détecté. Pensez à équilibrer vos données.")

    def fit(self, X, y, validation_data=None):
        """
        Entraîner le modèle CatBoost en utilisant la logique de boosting.
        
        Paramètres :
        - X : Caractéristiques d'entrée (features).
        - y : Étiquettes cibles.
        - validation_data : Tuple (X_val, y_val) pour la validation et l'arrêt anticipé.
        """
        # Encoder les caractéristiques catégoriques si nécessaire
        if isinstance(X, np.ndarray) and X.dtype.kind in 'OSU':  # Vérifier si X contient des chaînes ou catégories
            encoder = OrdinalEncoder()
            X = encoder.fit_transform(X)

        # Vérifier l'équilibre des classes
        self._check_balance(y)

        # Initialiser les prédictions de base (log-odds pour la classification binaire)
        pos_ratio = np.mean(y)
        self.base_prediction = np.log(pos_ratio / (1 - pos_ratio))
        predictions = np.full(len(y), self.base_prediction)

        # Initialiser le suivi des importances des caractéristiques
        self.feature_importances_ = np.zeros(X.shape[1])

        # Variables pour l'arrêt anticipé
        best_loss = float('inf')
        no_improvement_count = 0

        for iteration in range(self.n_estimators):
            # Calculer les résidus (gradients) avec la régularisation L2
            residuals = self._log_loss_gradient(y, predictions) - self.l2_reg * predictions

            # Entraîner un arbre de décision sur les résidus
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Stocker l'arbre entraîné
            self.trees.append(tree)

            # Mettre à jour les prédictions
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

            # Mettre à jour les importances des caractéristiques
            self.feature_importances_ += tree.feature_importances_

            # Vérifier la perte de validation pour l'arrêt anticipé
            if validation_data:
                X_val, y_val = validation_data
                val_predictions = self.predict_proba(X_val)
                val_loss = -np.mean(
                    y_val * np.log(val_predictions + 1e-9) + (1 - y_val) * np.log(1 - val_predictions + 1e-9)
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Arrêt anticipé si aucune amélioration n'est constatée
                if no_improvement_count >= self.early_stopping_rounds:
                    print(f"Arrêt anticipé à l'itération {iteration}.")
                    break

        # Normaliser les importances des caractéristiques
        self.feature_importances_ /= len(self.trees)

    def predict_proba(self, X):
        """
        Prédire les probabilités pour la classification binaire.
        
        Paramètres :
        - X : Caractéristiques d'entrée.
        
        Retourne :
        - Probabilités pour la classe positive.
        """
        predictions = np.full(len(X), self.base_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-predictions))  # Appliquer la fonction sigmoid

    def predict(self, X):
        """
        Faire des prédictions binaires.
        
        Paramètres :
        - X : Caractéristiques d'entrée.
        
        Retourne :
        - Prédictions binaires (0 ou 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
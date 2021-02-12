from sklearn.metrics import silhouette_score, calinski_harabasz_score


def eval_scores(data, labels):
    print("Silhouette score:", silhouette_score(data, labels, metric='cosine'))
    print("Calinski Harabasz score:", calinski_harabasz_score(data, labels))
